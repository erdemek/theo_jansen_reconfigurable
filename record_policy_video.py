from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from jansen_world import JansenEnv


def apply_camera(cam: mujoco.MjvCamera, lookat_x: float, lookat_y: float, lookat_z: float, azimuth: float, elevation: float, distance: float) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = [lookat_x, lookat_y, lookat_z]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a real-time-speed video of a trained Jansen PPO policy.")
    parser.add_argument("--model", default="jansen_reverted_reward_75M", help="Model path without .zip")
    parser.add_argument("--xml", default="jansen_assembly_red_articulated.xml", help="MuJoCo XML path")
    parser.add_argument("--out", default="videos/jansen_reverted_reward_75M_flat_realtime.mp4")
    parser.add_argument("--duration", type=float, default=10.0, help="Video duration in simulated seconds")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--cam-elevation", type=float, default=-14.0)
    parser.add_argument("--cam-distance", type=float, default=0.70)
    parser.add_argument("--cam-lookat-y", type=float, default=0.0)
    parser.add_argument("--cam-lookat-z", type=float, default=0.10)
    parser.add_argument("--follow", action="store_true", default=True, help="Follow the plate in x while recording")
    parser.add_argument("--no-follow", dest="follow", action="store_false")
    parser.add_argument("--reset-settle-time", type=float, default=6.2)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = JansenEnv(xml_path=args.xml, render_mode=None, reset_settle_time=args.reset_settle_time)
    obs, _ = env.reset()

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_frames = int(round(args.duration * args.fps))
    start_time = float(env.data.time)
    terminated = False
    truncated = False

    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame_idx in range(total_frames):
            target_elapsed = frame_idx / float(args.fps)
            while float(env.data.time) - start_time < target_elapsed and not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

            plate_x = float(env.data.qpos[env.plate_qadr])
            lookat_x = plate_x if args.follow else 0.0
            apply_camera(
                cam,
                lookat_x=lookat_x,
                lookat_y=args.cam_lookat_y,
                lookat_z=args.cam_lookat_z,
                azimuth=args.cam_azimuth,
                elevation=args.cam_elevation,
                distance=args.cam_distance,
            )
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            writer.append_data(np.asarray(frame))

            if terminated or truncated:
                break

    renderer.close()
    env.close()

    sim_seconds = max(0.0, float(env.data.time) - start_time)
    print(f"Video saved: {args.out}")
    print(f"fps={args.fps} | requested_duration={args.duration:.3f}s | simulated={sim_seconds:.3f}s | frames={frame_idx + 1}")
    print(f"terminated={terminated} | truncated={truncated}")


if __name__ == "__main__":
    main()
