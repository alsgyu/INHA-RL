# SIRL World-Model Learner

This branch adds a PPO-free learner for `K1/VelocityCommandWalk`.

## Training

```bash
python train_sirl_worldmodel.py \
  --task=K1/VelocityCommandWalk \
  --headless=True \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=2048
```

Do not validate an early unstable checkpoint with `--checkpoint -1` after a
failed run. Pass a known stable checkpoint explicitly, or move the failed log
directory out of `logs/` before using `-1`.

Use `--num_envs=1024` when tuning a new reward/config and `--num_envs=4096`
when the policy is stable enough to keep replay collection dense.

## MuJoCo Validation

The learner saves the actor under the same `model` checkpoint key used by the
existing MuJoCo script, so the usual path still works:

```bash
python play_mujoco_walk_getup_k1.py \
  --task K1/VelocityCommandWalk \
  --checkpoint -1 \
  --walk_only \
  --vx 0 \
  --vy 0 \
  --vyaw 0
```

Command-following checks:

```bash
python play_mujoco_walk_getup_k1.py \
  --task K1/VelocityCommandWalk \
  --checkpoint logs/K1/K1/VelocityCommandWalkSIRL/<run>/nn/model_500.pth \
  --walk_only \
  --vx 0.1 \
  --vy 0 \
  --vyaw 0 \
  --play_velocity_metrics
```

For this branch, check low-speed straight walking first. Validate `vx=0`,
then `vx=0.1`, then `vx=0.2`; do not use `vx=0.5` as the first pass/fail
test while the straight gait is still being tuned.

## 4090 Starting Point

- `num_envs`: 1024 for debugging, 2048 for default training, 4096 after stability.
- `batch_size`: 512.
- `updates_per_iter`: 16 with `collect_steps_per_iter=16`.
- Real replay: CPU, 1M transitions.
- Model replay: CPU, 250k transitions.
- World model ensemble: 5 MLPs, hidden dims `[512, 512]`.
- Model rollout horizon: 1.
- Model batch ratio: `0.0` until MuJoCo straight walking is stable; then try
  `0.02`, `0.05`, set `model_rollout_enabled: true`, and only increase if
  lateral drift does not grow.
- Keep `sirl_loss: mse` while logstd is clamped tightly; NLL can dominate the
  actor update when off-policy action targets are imperfect.

## Stability Notes

The current safe path is real replay + teacher bootstrap + return-filtered
trajectory SIRL + mirror symmetry. The world model still trains as an auxiliary
model, but generated transitions are not mixed into SAC updates by default.
Turn model-generated updates back on only after `vx=0.1` and `vx=0.2` hold a
straight path in MuJoCo.

## Extension Path

1. Replace observation-space one-step models with latent recurrent dynamics.
2. Add uncertainty penalties from ensemble disagreement.
3. Add CEM/MPPI action-sequence planning as a policy improvement target.
4. Move from behavior cloning on elite trajectories to lower-bound Q-learning over
   n-step returns.
5. Distill planned actions back into the deployable `BaseActorCritic.actor`.
