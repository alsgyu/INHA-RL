# FastSAC K1 Training

This branch adds a Booster K1 training path based on the FastSAC recipe from
`Learning Sim-to-Real Humanoid Locomotion in 15 Minutes` and the official
Holosoma implementation:

- Project page: https://younggyo.me/fastsac-humanoid/
- Code: https://github.com/amazon-far/holosoma

The implementation uses the existing K1 `VelocityCommandWalkK1` environment and
asset, while mapping the Holosoma FastSAC locomotion recipe onto it:
off-policy replay, a distributional double-Q critic, observation normalization,
LayerNorm/SiLU actor and critic networks, symmetry augmentation, tanh-bounded
actions, mixed terrain/randomization, a minimal paper-style reward set, and an
action-rate/tilt penalty curriculum.

## Train

```bash
python train_fastsac_k1.py --task=K1/VelocityCommandWalkFastSAC
```

Useful overrides:

```bash
python train_fastsac_k1.py \
  --task=K1/VelocityCommandWalkFastSAC \
  --num_envs=2048 \
  --max_iterations=50000
```

`K1/VelocityCommandWalkFastSAC` keeps the Booster K1 environment and asset but
uses the Holosoma FastSAC recipe as directly as this repository's env API
allows: 4096 envs, global batch 8192, 8 updates per step, compiled update
functions, mixed terrain, push-style velocity perturbations, action delay, PD
gain/mass/friction randomization, and a minimal paper-style reward mapping.

The default config targets one RTX 4090 style setup:

- `num_envs: 4096`
- `batch_size: 8192`
- `num_updates: 8`
- `gamma: 0.97`
- `tau: 0.125`
- `buffer_size: 1024` per environment

Training logs include:

- `perf/env_steps_per_sec`
- `perf/updates_per_sec`
- `perf/replay_ratio`
- `rollout/done_rate`
- `sac/q_loss`
- `sac/actor_loss`
- `sac/alpha`

These are the first metrics to inspect when training is much slower than the
paper baseline.

## Replay Preservation

FastSAC checkpoints are saved under:

```text
logs/K1/K1/VelocityCommandWalkFastSAC/<timestamp>/nn/
```

Replay snapshots are optional and are off by default to avoid IO overhead during
paper-style fast runs. If `algorithm.fast_sac.save_replay_interval` is enabled,
snapshots are saved under:

```text
logs/K1/K1/VelocityCommandWalkFastSAC/<timestamp>/replay/
```

To warm start from preserved transitions:

```bash
python train_fastsac_k1.py \
  --task=K1/VelocityCommandWalkFastSAC \
  --replay_path=logs/K1/K1/VelocityCommandWalkFastSAC/<run>/replay/replay_5000.npz
```

The preload path accepts `.npz` files containing at least:

- `obs`
- `action` or `actions`
- `reward`
- `done`
- `next_obs`

If `critic_obs` and `next_critic_obs` are missing, they are padded from public
observations so old transition data can still be used.

## Optional Teacher Distillation

The runner can imitate an existing TorchScript policy during early training:

```bash
python train_fastsac_k1.py \
  --task=K1/VelocityCommandWalkFastSAC \
  --teacher_policy_path=deploy/models/velocity_command_walk_k1.pt
```

Enable the loss by setting `algorithm.fast_sac.teacher_bc_coef` in the YAML.
The default is `0.0`, so the paper-style FastSAC path is the default.

## Export For K1 Deployment

After training:

```bash
python export_model.py --task=K1/VelocityCommandWalkFastSAC --checkpoint=-1
```

For FastSAC checkpoints, `export_model.py` exports a TorchScript wrapper that
applies the saved observation normalizer and then calls the actor. The exported
policy keeps the same deployment interface as existing policies:

```text
obs -> action
```

The config metadata sets:

```text
policy_name: velocity_command_walk_k1_fastsac
```

so export copies the final artifacts into `deploy/models/`.
