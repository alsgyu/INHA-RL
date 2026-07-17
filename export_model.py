import os
import glob
import json
import shutil
import yaml
import argparse
import torch
from utils.models.BaseAC import *


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    cfg = load_config(cfg_file)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = BaseActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
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
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    model.eval()
    script_module = torch.jit.script(model.actor)
    save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + ".pt"
    script_module.save(save_path)
    print(f"Saved model to {save_path}")

    metadata = cfg.get("metadata", {}).copy()
    metadata["task"] = args.task
    metadata["model_class"] = cfg.get("basic", {}).get("model", "BaseActorCritic")
    metadata["num_actions"] = cfg["env"]["num_actions"]
    metadata["num_observations"] = cfg["env"]["num_observations"]
    metadata["num_privileged_obs"] = cfg["env"]["num_privileged_obs"]
    metadata["normalization"] = cfg.get("normalization", {})
    if "commands" in cfg:
        metadata["commands"] = cfg["commands"]
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
