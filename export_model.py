import os
import glob
import hashlib
import json
import shutil
import yaml
import argparse
import torch
from utils.models.BaseAC import *
from utils.fast_sac import EmpiricalNormalizer, FastSACActor, FastSACPolicyWrapper


def merge_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(cfg_file, visited=None):
    if visited is None:
        visited = set()
    cfg_file = os.path.normpath(cfg_file)
    if cfg_file in visited:
        raise ValueError(f"Recursive config inheritance detected for {cfg_file}")
    visited.add(cfg_file)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    parent = cfg.pop("extends", None)
    if not parent:
        return cfg
    if not parent.endswith(".yaml"):
        parent = os.path.join("envs", f"{parent}.yaml")
    elif not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(cfg_file), parent)
    return merge_dicts(load_config(parent, visited), cfg)


def get_robot_type(task_name):
    """Determine robot type from task name."""
    # Check if task name starts with K1 or T1
    if task_name.startswith("K1"):
        return "K1"
    elif task_name.startswith("T1"):
        return "T1"
    else:
        # Default fallback - could be extended for other robot types
        return "Unknown"


def file_sha1(path):
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def action_scale_from_cfg(cfg):
    sac_cfg = cfg.get("algorithm", {}).get("fast_sac", {})
    configured = sac_cfg.get("action_scale")
    num_actions = cfg["env"]["num_actions"]
    if configured is not None:
        if isinstance(configured, (list, tuple)):
            return torch.tensor(configured, dtype=torch.float)
        return torch.full((num_actions,), float(configured), dtype=torch.float)
    clip_actions = float(cfg.get("normalization", {}).get("clip_actions", 1.0))
    return torch.full((num_actions,), clip_actions, dtype=torch.float)


def export_fast_sac(model_dict, cfg, checkpoint_path):
    metadata_cfg = model_dict.get("task_cfg") if isinstance(model_dict.get("task_cfg"), dict) else cfg
    sac_cfg = metadata_cfg.get("algorithm", {}).get("fast_sac", {})

    actor_state = model_dict["actor_state_dict"]
    # Detect old-format checkpoint (fc_mu was Sequential, had action_bias)
    is_old_format = "fc_mu.0.weight" in actor_state
    if is_old_format:
        remapped = {}
        for key, value in actor_state.items():
            if key == "fc_mu.0.weight":
                remapped["fc_mu.weight"] = value
            elif key == "fc_mu.0.bias":
                pass  # old inner bias, discard
            elif key == "fc_logstd.0.weight":
                remapped["fc_logstd.weight"] = value
            elif key == "fc_logstd.0.bias":
                remapped["fc_logstd.bias"] = value
            elif key == "action_bias":
                remapped["fc_mu.bias"] = value
            else:
                remapped[key] = value
        actor_state = remapped

    # Determine actual obs_dim from checkpoint weights
    net0_weight = actor_state.get("net.0.weight")
    obs_dim = net0_weight.shape[1] if net0_weight is not None else metadata_cfg["env"]["num_observations"]

    actor = FastSACActor(
        obs_dim,
        metadata_cfg["env"]["num_actions"],
        hidden_dim=int(sac_cfg.get("actor_hidden_dim", 512)),
        log_std_min=float(sac_cfg.get("log_std_min", -5.0)),
        log_std_max=float(sac_cfg.get("log_std_max", 0.0)),
        use_tanh=bool(sac_cfg.get("use_tanh", True)),
        use_layer_norm=bool(sac_cfg.get("use_layer_norm", True)),
        action_scale=action_scale_from_cfg(metadata_cfg),
        device="cpu",
    )
    actor.load_state_dict(actor_state, strict=False)

    obs_normalizer = None
    if bool(sac_cfg.get("obs_normalization", True)) and model_dict.get("obs_normalizer_state") is not None:
        obs_normalizer = EmpiricalNormalizer(obs_dim, "cpu")
        obs_normalizer.load_state_dict(model_dict["obs_normalizer_state"])
    wrapper = FastSACPolicyWrapper(actor, obs_normalizer)
    wrapper.eval()
    save_path = os.path.splitext(checkpoint_path)[0] + ".pt"
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    script_module = torch.jit.trace(wrapper, dummy_obs)
    script_module.save(save_path)
    print(f"Saved FastSAC policy to {save_path}")
    return save_path, metadata_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    cfg = load_config(cfg_file)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
        task_name = cfg["basic"].get("log_task", args.task)
        robot_type = get_robot_type(task_name)
        
        # First try: exact log task in robot-specific folder
        search_task_names = [task_name]
        for fallback_name in (cfg["basic"].get("task"), args.task):
            if fallback_name and fallback_name not in search_task_names:
                search_task_names.append(fallback_name)

        task_models = []
        for search_task_name in search_task_names:
            task_log_pattern = os.path.join("logs", robot_type, search_task_name, "**/*.pth")
            task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
            if task_models:
                break
        
        if task_models:
            cfg["basic"]["checkpoint"] = task_models[-1]
        else:
            # Second try: any task in robot-specific folder
            robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
            robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
            
            if robot_models:
                cfg["basic"]["checkpoint"] = robot_models[-1]
            else:
                # Fallback: all logs if no robot-specific models found
                cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch_load_checkpoint(cfg["basic"]["checkpoint"], map_location="cpu")
    fast_sac_checkpoint = bool(model_dict.get("fast_sac", False) or "actor_state_dict" in model_dict)
    if fast_sac_checkpoint:
        save_path, metadata_cfg = export_fast_sac(model_dict, cfg, cfg["basic"]["checkpoint"])
    else:
        metadata_cfg = model_dict.get("task_cfg")
        metadata_cfg = metadata_cfg if isinstance(metadata_cfg, dict) else cfg
        model = BaseActorCritic(
            metadata_cfg["env"]["num_actions"],
            metadata_cfg["env"]["num_observations"],
            metadata_cfg["env"]["num_privileged_obs"],
        )
        model.load_state_dict(model_dict["model"])

        model.eval()
        script_module = torch.jit.script(model.actor)
        save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + ".pt"
        script_module.save(save_path)
        print(f"Saved model to {save_path}")

    metadata = metadata_cfg.get("metadata", {}).copy()
    metadata["task"] = args.task
    metadata["checkpoint_path"] = cfg["basic"]["checkpoint"]
    metadata["checkpoint_sha1"] = file_sha1(cfg["basic"]["checkpoint"])
    metadata["exported_policy_path"] = save_path
    metadata["exported_policy_sha1"] = file_sha1(save_path)
    metadata["model_class"] = "FastSACActor" if fast_sac_checkpoint else metadata_cfg.get("basic", {}).get("model", "BaseActorCritic")
    metadata["algorithm"] = "fast_sac" if fast_sac_checkpoint else "ppo"
    if fast_sac_checkpoint:
        metadata["global_step"] = int(model_dict.get("global_step", 0))
    metadata["num_actions"] = metadata_cfg["env"]["num_actions"]
    metadata["num_observations"] = metadata_cfg["env"]["num_observations"]
    metadata["num_privileged_obs"] = metadata_cfg["env"]["num_privileged_obs"]
    metadata["normalization"] = metadata_cfg.get("normalization", {})
    metadata["control"] = metadata_cfg.get("control", {})
    metadata["init_state"] = metadata_cfg.get("init_state", {})
    if "commands" in metadata_cfg:
        metadata["commands"] = metadata_cfg["commands"]
    metadata_path = os.path.splitext(save_path)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    policy_name = metadata.get("policy_name")
    if policy_name:
        deploy_model_dir = os.path.join("deploy", "models")
        os.makedirs(deploy_model_dir, exist_ok=True)
        deploy_model_path = os.path.join(deploy_model_dir, f"{policy_name}.pt")
        deploy_metadata_path = os.path.join(deploy_model_dir, f"{policy_name}.metadata.json")
        shutil.copy2(save_path, deploy_model_path)
        shutil.copy2(metadata_path, deploy_metadata_path)
        print(f"Copied deploy model to {deploy_model_path}")
        print(f"Copied deploy metadata to {deploy_metadata_path}")
