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
  --checkpoint logs/K1/K1/VelocityCommandWalkSIRL/<run>/nn/model_2000.pth \
  --walk_only \
  --vx 0.1 \
  --vy 0 \
  --vyaw 0 \
  --play_velocity_metrics
```

For this branch, check low-speed straight walking first. Validate `vx=0`,
then `vx=0.1`, then `vx=0.2`; do not use `vx=0.5` as the first pass/fail
test while the straight gait is still being tuned. Early checkpoints are now
teacher-distilled, but prefer `model_2000.pth` or later for MuJoCo checks.

For a checkpoint that already walks well but drifts slightly during straight
commands, keep the policy weights unchanged and enable the MuJoCo-only straight
path correction:

```bash
python play_mujoco_walk_getup_k1.py \
  --task K1/VelocityCommandWalk \
  --checkpoint logs/K1/K1/VelocityCommandWalkSIRL/<run>/nn/model_30000.pth \
  --walk_only \
  --vx 1.5 --vy 0 --vyaw 0 \
  --duration_s 10 \
  --straight_path_correction
```

Do not use this correction as proof of deploy readiness. It changes the command
sent into the policy at play time and is useful only as a diagnostic or as a
high-level controller prototype.

## Command Fine-Tune

To improve the policy weights themselves, fine-tune from the known-good
checkpoint with the command-tracking config. The checkpoint is also loaded as
the teacher anchor, so the new actor is penalized if it drifts too far from the
walking behavior that already works.

```bash
python train_sirl_worldmodel.py \
  --task=K1/VelocityCommandWalkCmdFineTune \
  --checkpoint logs/K1/K1/VelocityCommandWalkSIRL/2026-07-26-21-35-34/nn/model_30000.pth \
  --headless=True \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=2048 \
  --max_iterations=3000
```

Validate the fine-tuned checkpoint without MuJoCo straight-path correction:

```bash
python play_mujoco_walk_getup_k1.py \
  --task K1/VelocityCommandWalk \
  --checkpoint logs/K1/K1/VelocityCommandWalkSIRLCmdFineTune/<run>/nn/model_500.pth \
  --walk_only \
  --vx 1.5 --vy 0 --vyaw 0 \
  --duration_s 10
```

Keep `2026-07-26-21-35-34/nn/model_30000.pth` as the golden rollback
checkpoint. Check `model_500.pth`, `model_1000.pth`, and `model_2000.pth`
before considering the final checkpoint. Stop the fine-tune if `vx=0.1`,
`vx=0.5`, or `vx=1.5` becomes less stable than the golden checkpoint.

The fine-tune runner resets optimizer, critic, alpha, and world-model state from
the checkpoint by default for this config. It keeps only the policy weights and
uses the same checkpoint as a teacher anchor with a rollback guard, so a single
bad actor update cannot move the deploy policy far from the known-good gait.

If the policy walks stably but arcs to one side, look for swing-leg asymmetry:
for example, one knee or foot yawing outward during swing. The command fine-tune
config includes straight-only penalties for swing foot yaw, swing lateral foot
velocity, and swing roll/yaw actions to reduce that kind of repeated side
impulse without adding any MuJoCo-only command correction.

## 4090 Starting Point

- `num_envs`: 1024 for debugging, 2048 for default training, 4096 after stability.
- `batch_size`: 512.
- `updates_per_iter`: 2 with `collect_steps_per_iter=16`.
- Teacher distillation: 6 batches of 4096 samples per iteration for the full
  30k-iteration warmup so the saved student actor stays close to the deploy
  teacher.
- SAC actor updates: disabled by default. Keep Q/world-model training auxiliary
  until MuJoCo validation is no worse than the teacher, then re-enable actor RL
  with a tiny coefficient such as `actor_rl_coef: 0.02`.
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

The current safe path is real replay + teacher bootstrap + supervised actor
distillation. The world model, critic, and SIRL buffers still train as
auxiliary components, but they do not directly move the deploy actor by default.
Turn actor RL, SIRL actor loss, symmetry loss, or model-generated updates back
on only after `vx=0.1` and `vx=0.2` hold a straight path in MuJoCo.

## Extension Path

1. Replace observation-space one-step models with latent recurrent dynamics.
2. Add uncertainty penalties from ensemble disagreement.
3. Add CEM/MPPI action-sequence planning as a policy improvement target.
4. Move from behavior cloning on elite trajectories to lower-bound Q-learning over
   n-step returns.
5. Distill planned actions back into the deployable `BaseActorCritic.actor`.
