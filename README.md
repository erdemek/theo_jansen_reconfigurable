# Reconfigurable Theo Jansen MuJoCo Walker

MuJoCo and reinforcement-learning workspace for a reconfigurable Theo Jansen-style walking mechanism. The model uses multiple articulated legs with controllable prismatic link targets for `k`, `f`, and `h` links.

The current default flat-ground model is:

```text
jansen_assembly_red_articulated.xml
```

The default contact friction in that XML is:

```xml
friction="0.6 0.08 0.01"
```

where MuJoCo interprets the values as sliding, torsional, and rolling friction.

## Main Files

```text
jansen_world.py                    Gymnasium/MuJoCo RL environment
train_jansen.py                    PPO train, eval, analyze, and eval-video CLI
run_jansen_prismatic_phase.py      Manual grouped k/f/h prismatic phase gait runner
run_jansen_assembly.py             Manual assembly simulation runner
record_policy_video.py             Standalone policy video recorder
```

Terrain generators and example terrain XMLs:

```text
generate_rolling_hills_terrain.py
jansen_assembly_rolling_hills.xml
jansen_assembly_rolling_hills_stronger.xml
jansen_assembly_farm_field.xml
```

## Reward

The current training reward in `jansen_world.py` is:

```text
reward =
    1000 * (x_before - x_after)
  - 10 * |vz|
  - 5  * |vy|
  - 20 * (1 - up_z)
  - 30 * |up_y|
  - 4  * (|wx| + |wy|)
  - 2  * |heading_y|
  - 10 * sum(|prismatic_target - previous_prismatic_target|)
  + 0.5
```

Fall termination:

```text
up_z < 0.6 or body_z < 0.05
```

## Training

Start a fresh 30M-step training run:

```powershell
python train_jansen.py --mode train --model jansen_reverted_reward_30M --steps 30000000 --envs 6
```

Continue from a saved model:

```powershell
python train_jansen.py --mode train --model jansen_reverted_reward_60M --load jansen_reverted_reward_30M --steps 30000000 --envs 6
```

The model name is given without `.zip`.

## Evaluation

Open the MuJoCo viewer:

```powershell
python train_jansen.py --mode eval --model jansen_reverted_reward_75M
```

Record a real-time-speed eval video on flat ground:

```powershell
python train_jansen.py --mode eval --model jansen_reverted_reward_75M --video videos/jansen_reverted_reward_75M_eval_flat_realtime.mp4 --video-duration 10 --video-fps 60 --video-width 640 --video-height 480
```

Behind-view video:

```powershell
python train_jansen.py --mode eval --model jansen_reverted_reward_75M --video videos/jansen_reverted_reward_75M_eval_flat_realtime_behind.mp4 --video-duration 10 --video-fps 60 --video-width 640 --video-height 480 --video-cam-azimuth 180 --video-cam-elevation -14 --video-cam-distance 0.70
```

## Notes

Training outputs are intentionally not tracked:

```text
*.zip
logs/
models/
tensorboard_logs/
videos/
plots/
terrain_assets/
```

This keeps the repository focused on source scripts and MuJoCo XML files.
