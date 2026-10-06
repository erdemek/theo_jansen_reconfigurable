# Reconfigurable Theo Jansen MuJoCo Walker

MuJoCo and reinforcement-learning workspace for a reconfigurable Theo Jansen-style walker. Four crank motors provide the periodic drive, while PPO can control selected `k`, `f`, and `h` prismatic link targets independently for each leg.

## Main files

```text
jansen_world.py                         Gymnasium/MuJoCo RL environment
train_jansen.py                         PPO train, eval, analyze, checkpoint, and video CLI
run_jansen_assembly.py                  Manual assembly simulation and video runner
generate_rolling_hills_terrain.py       Seeded continuous-heightfield generator
plot_reward_terms.py                    TensorBoard reward decomposition
plot_crank_rotations.py                 Crank angle and velocity diagnostics
plot_prismatic_command_response.py      Command/response, rate, and acceleration diagnostics
```

Current model variants:

```text
jansen_assembly.xml
jansen_assembly_red_articulated.xml
jansen_assembly_rolling_hills.xml
jansen_assembly_rolling_hills_torso15cm.xml
jansen_assembly_rolling_hills_torso15cm_red_ground.xml
```

The rolling-hills XMLs use the tracked `terrain_assets/rolling_hills_heightmap.png`. Regenerate it and the default rolling-hills XML with:

```powershell
python generate_rolling_hills_terrain.py
```

## Environment

The observation has 29 values:

```text
root linear/angular velocity (6)
body-up vector (3)
four crank phases as sin/cos (8)
k/f/h prismatic states (12)
```

The action contains four values for every selected prismatic group. For example, `--train-prismatic-groups h` produces a four-dimensional action space, while `kfh` produces twelve actions. Commands are rate-limited, with optional acceleration limiting.

The default `h_only_straight_walk` reward is:

```text
reward =
    2000 * (x_after - x_before)
  + 5 * vx
  - 2 * (|wx| + |wy|)
  - 6 * |up_y|
  + 0.5
```

Additional reward presets are available through `--reward-preset`. Episodes terminate when `up_z < 0.6` or torso height falls below `0.05 m`, and truncate after 10,000 control steps.

## Training

Example `h`-only flat-ground run:

```powershell
python train_jansen.py --mode train --model jansen_h_only_30M --steps 30000000 --envs 6 --train-prismatic-groups h
```

Example rolling-hills run with wider `h` motion and acceleration limiting:

```powershell
python train_jansen.py --mode train --model jansen_hills_h50mm_60M --steps 60000000 --envs 6 --xml jansen_assembly_rolling_hills_torso15cm.xml --train-prismatic-groups h --h-positive-amplitude 0.05 --h-negative-amplitude 0.05 --prismatic-max-acceleration 0.5 --reward-preset h_only_straight_walk_normalized_fullscale
```

Resume from a checkpoint or final model:

```powershell
python train_jansen.py --mode train --model jansen_h_only_60M --load jansen_h_only_30M --steps 30000000 --envs 6 --train-prismatic-groups h
```

Environment arguments used for evaluation must match those used to train the checkpoint, especially the selected prismatic groups and amplitude limits.

## Evaluation

Open the viewer:

```powershell
python train_jansen.py --mode eval --model jansen_h_only_30M --train-prismatic-groups h
```

Record a video:

```powershell
python train_jansen.py --mode eval --model jansen_h_only_30M --train-prismatic-groups h --video videos/jansen_h_only_eval.mp4 --video-duration 10
```

Report deterministic episode rewards:

```powershell
python train_jansen.py --mode analyze --model jansen_h_only_30M --train-prismatic-groups h
```

## Repository contents

Source, XML models, and the required rolling-hills heightmap are tracked. Generated model weights, checkpoints, logs, TensorBoard events, plots, videos, and local experiment bundles remain untracked.
