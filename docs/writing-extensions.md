# Adding Config Values to Holosoma

Two ways to add a preset (robot, reward, simulator, inference policy, …): a **`--import-file`** for local/one-off use, or a packaged **entry point** for anything you ship. Both land in the same menu and work across `run_sim`, `train_agent`, `eval_agent`, `replay`, `run_policy`.

Each config family is a `ConfigRegistry` with an entry-point group. Presets are type-checked on the way in; a wrong type or a broken plugin is skipped.

## `--import-file` — no packaging

A plain `.py` file that calls `REGISTRY.add(...)`:

```python
# my_presets.py  (anywhere on disk)
from dataclasses import replace
from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof

ROBOT_REGISTRY.add("g1_stiff", replace(g1_29dof, control=replace(g1_29dof.control, action_scale=0.1)))
```

```bash
python -m holosoma.run_sim --import-file my_presets.py robot:g1-stiff   # repeatable; selectable like a built-in
```

`.add(name, value)` type-checks `value` and returns it, so `x = REGISTRY.add(...)` also keeps a normal module attribute.

## Entry point — packaged extension

The value must be a config instance of the family's type. Compose higher-level presets from lower-level ones:

```python
# holosoma_inference_ext_quadruped/config_values/robot.py
from holosoma_inference.config.config_types.robot import RobotConfig
go2_12dof = RobotConfig(robot_type="go2_12dof", robot="go2", num_motors=12, num_joints=12)  # ...

# holosoma_inference_ext_quadruped/config_values/inference.py
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import task          # reuse core presets
from holosoma_inference_ext_quadruped.config_values import observation, robot
go2_12dof_loco = InferenceConfig(robot=robot.go2_12dof, observation=observation.loco_go2_12dof, task=task.locomotion)
```

Declare one entry point per preset — `<name> = "<module>:<attr>"`, group picks the registry:

```toml
# pyproject.toml
[project]
dependencies = ["holosoma_inference"]        # or "holosoma" for training-side presets
[tool.setuptools.packages.find]
include = ["holosoma_inference_ext_quadruped*"]   # ship your config_values modules

[project.entry-points."holosoma.config.robot"]
go2-12dof = "holosoma_inference_ext_quadruped.config_values.robot:go2_12dof"
[project.entry-points."holosoma.config.inference"]
go2-12dof-loco = "holosoma_inference_ext_quadruped.config_values.inference:go2_12dof_loco"
```

```bash
pip install -e .
python -m holosoma_inference.run_policy inference:go2-12dof-loco   # discovered automatically, no registration code
```

## Adding a preset in the core repo

Edit the family's `config_values` module and register with `.add()`:

```python
# holosoma/config_values/robot.py
my_robot = ROBOT_REGISTRY.add("my_robot", RobotConfig(...))
```

## Naming

Register hyphen-case (`go2-12dof`). The CLI token is `<field>:<key>` and accepts both forms — `robot:g1_29dof` and `robot:g1-29dof` both work.

## Config families

Publish an entry point under the group whose config type matches your preset. Training and inference share some group names on purpose — the type check routes each preset to the right registry.

**Training (`holosoma`)** — `robot` `RobotConfig` · `simulator` `SimulatorConfig` · `run_sim` `SimulatorConfig` · `terrain` `TerrainManagerCfg` · `scene` `SceneConfig` · `algo` `PPOAlgoConfig`/`FastSACAlgoConfig` · `observation` `ObservationManagerCfg` · `action` `ActionManagerCfg` · `reward` `RewardManagerCfg` · `termination` `TerminationManagerCfg` · `randomization` `RandomizationManagerCfg` · `command` `CommandManagerCfg` · `curriculum` `CurriculumManagerCfg` · `logger` `DisabledLoggerConfig`/`WandbLoggerConfig` · `experiment` `ExperimentConfig` (top-level `exp:`)

**Inference (`holosoma_inference`)** — `robot` `RobotConfig` · `observation` `ObservationConfig` · `task` `TaskConfig` · `inference` `InferenceConfig` (top-level `inference:`)

Group string is `holosoma.config.<family>`, registry var is `<FAMILY>_REGISTRY` (e.g. `holosoma.config.reward` → `REWARD_REGISTRY`).

## Don't

```python
# ❌ Snapshot merge — misses presets registered later; visible only via this module.
DEFAULTS = {**CORE_DEFAULTS, "x1_25dof": x1_25dof}
# ❌ In-place mutation — import-side-effect global; bypasses the type check.
CORE_DEFAULTS.update({"elf3_29dof": elf3_29dof})
# ❌ Hand-rolled ep loop — no error isolation; one bad plugin crashes the CLI.
for ep in entry_points(group="holosoma.config.inference"): all_defaults[ep.name] = ep.load()
```

`module.DEFAULTS` / `get_defaults()` still work but warn. Use an entry point, or `.add()` / `--import-file`.
