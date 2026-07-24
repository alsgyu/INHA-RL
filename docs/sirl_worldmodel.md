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
  --checkpoint -1 \
  --walk_only \
  --vx 0.3 \
  --vy 0 \
  --vyaw 0 \
  --play_velocity_metrics
```

## 4090 Starting Point

- `num_envs`: 1024 for debugging, 2048 for default training, 4096 after stability.
- `batch_size`: 1024.
- `updates_per_iter`: 64 with `collect_steps_per_iter=16`.
- Real replay: CPU, 1M transitions.
- Model replay: CPU, 250k transitions.
- World model ensemble: 5 MLPs, hidden dims `[512, 512]`.
- Model rollout horizon: 1 at first, then 2-3 after prediction error stabilizes.
- Model batch ratio: 0.25 at first, up to 0.5 only if MuJoCo validation remains stable.

## Extension Path

1. Replace observation-space one-step models with latent recurrent dynamics.
2. Add uncertainty penalties from ensemble disagreement.
3. Add CEM/MPPI action-sequence planning as a policy improvement target.
4. Move from behavior cloning on elite trajectories to lower-bound Q-learning over
   n-step returns.
5. Distill planned actions back into the deployable `BaseActorCritic.actor`.
