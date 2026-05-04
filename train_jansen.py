import os
import argparse
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from collections import defaultdict, deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from jansen_world import JansenEnv


class RewardTermsTensorboardCallback(BaseCallback):
    def __init__(self, window_size=100):
        super().__init__()
        self.window_size = window_size
        self.episode_terms = defaultdict(lambda: deque(maxlen=self.window_size))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            terms = info.get("episode_reward_terms")
            if terms is None:
                continue
            for name, value in terms.items():
                self.episode_terms[name].append(float(value))

        for name, values in self.episode_terms.items():
            if values:
                self.logger.record(
                    f"rollout_reward_terms/{name}_mean",
                    float(np.mean(values)),
                )
        return True


def make_env(
    render=False,
    monitor_path=None,
    xml_path="jansen_assembly_red_articulated.xml",
    prismatic_cmd_limit=0.005,
    k_positive_amplitude=0.01,
    f_positive_amplitude=0.002,
    h_positive_amplitude=0.01,
    k_negative_amplitude=0.0,
    f_negative_amplitude=0.0,
    h_negative_amplitude=0.01,
    slew_rate=0.05,
    reset_settle_time=6.2,
    train_prismatic_groups="kfh",
):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        render_mode = "human" if render else None
        env = JansenEnv(
            render_mode=render_mode,
            xml_path=xml_path,
            prismatic_cmd_limit=prismatic_cmd_limit,
            k_positive_amplitude=k_positive_amplitude,
            f_positive_amplitude=f_positive_amplitude,
            h_positive_amplitude=h_positive_amplitude,
            k_negative_amplitude=k_negative_amplitude,
            f_negative_amplitude=f_negative_amplitude,
            h_negative_amplitude=h_negative_amplitude,
            k_slew_rate=slew_rate,
            reset_settle_time=reset_settle_time,
            train_prismatic_groups=train_prismatic_groups,
        )
        if monitor_path:
            env = Monitor(env, monitor_path)
        else:
            env = Monitor(env)
        return env
    return _init

def train(
    timesteps=2000000,
    model_name="jansen_rl_expert_2M",
    load_model=None,
    n_envs=6,
    render=False,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gpu_heavy=False,
    xml_path="jansen_assembly_red_articulated.xml",
    device_preference="cpu",
    prismatic_cmd_limit=0.005,
    k_positive_amplitude=0.01,
    f_positive_amplitude=0.002,
    h_positive_amplitude=0.01,
    k_negative_amplitude=0.0,
    f_negative_amplitude=0.0,
    h_negative_amplitude=0.01,
    slew_rate=0.05,
    reset_settle_time=6.2,
    train_prismatic_groups="kfh",
):
    """
    Train the Jansen Walker using 6 Parallel Environments and GPU.
    """
    if device_preference == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_preference

    # Optional GPU-focused profile: bigger model + bigger batches.
    policy_kwargs = {}
    if gpu_heavy and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        n_steps = max(n_steps, 4096)
        batch_size = max(batch_size, 2048)
        n_epochs = max(n_epochs, 20)
        policy_kwargs = dict(net_arch=[512, 512, 256])

    rollout_size = n_steps * n_envs
    if batch_size > rollout_size:
        batch_size = rollout_size

    print(f"Starting Multi-Env Training on {device.upper()}")
    print(f"Parameters: {timesteps} steps | {n_envs} parallel envs | -> {model_name}.zip")
    if load_model:
        print(f"Resume from: {load_model}.zip")
    print(
        f"PPO: n_steps={n_steps} | batch_size={batch_size} | n_epochs={n_epochs} | "
        f"rollout={rollout_size} | gpu_heavy={gpu_heavy and device == 'cuda'}"
    )
    print(
        "Amplitude limits: "
        f"k=[-{k_negative_amplitude}, +{k_positive_amplitude}] m | "
        f"f=[-{f_negative_amplitude}, +{f_positive_amplitude}] m | "
        f"h=[-{h_negative_amplitude}, +{h_positive_amplitude}] m"
    )
    print(f"Train prismatic groups: {train_prismatic_groups}")
    
    # Create parallel environments
    # Note: only the first env can be rendered easily in some setups, 
    # but here we usually train headless (render=False)
    try:
        env = SubprocVecEnv([
            make_env(
                render=False,
                xml_path=xml_path,
                prismatic_cmd_limit=prismatic_cmd_limit,
                k_positive_amplitude=k_positive_amplitude,
                f_positive_amplitude=f_positive_amplitude,
                h_positive_amplitude=h_positive_amplitude,
                k_negative_amplitude=k_negative_amplitude,
                f_negative_amplitude=f_negative_amplitude,
                h_negative_amplitude=h_negative_amplitude,
                slew_rate=slew_rate,
                reset_settle_time=reset_settle_time,
                train_prismatic_groups=train_prismatic_groups,
            )
            for _ in range(n_envs)
        ])
    except PermissionError as e:
        print(f"SubprocVecEnv unavailable ({e}). Falling back to DummyVecEnv.")
        env = DummyVecEnv([
            make_env(
                render=False,
                xml_path=xml_path,
                prismatic_cmd_limit=prismatic_cmd_limit,
                k_positive_amplitude=k_positive_amplitude,
                f_positive_amplitude=f_positive_amplitude,
                h_positive_amplitude=h_positive_amplitude,
                k_negative_amplitude=k_negative_amplitude,
                f_negative_amplitude=f_negative_amplitude,
                h_negative_amplitude=h_negative_amplitude,
                slew_rate=slew_rate,
                reset_settle_time=reset_settle_time,
                train_prismatic_groups=train_prismatic_groups,
            )
            for _ in range(n_envs)
        ])
    
    if load_model:
        model = PPO.load(load_model, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=0.01,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs/"
        )

    print(f"Learning... (Running on {device} with {n_envs} workers)")
    try:
        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            reset_num_timesteps=not bool(load_model),
            callback=RewardTermsTensorboardCallback(),
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving progress...")
        
    model.save(model_name)
    env.close()
    print(f"Training Complete. Model saved as {model_name}.zip")

def analyze(
    model_path="jansen_rl_expert_2M",
    episodes=20,
    xml_path="jansen_assembly_red_articulated.xml",
    prismatic_cmd_limit=0.005,
    k_positive_amplitude=0.01,
    f_positive_amplitude=0.002,
    h_positive_amplitude=0.01,
    k_negative_amplitude=0.0,
    f_negative_amplitude=0.0,
    h_negative_amplitude=0.01,
    slew_rate=0.05,
    reset_settle_time=6.2,
    train_prismatic_groups="kfh",
):
    """
    Evaluate the model over N episodes and report the Best and Worst rewards.
    """
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        return

    print(f"Analyzing {model_path} over {episodes} episodes...")
    model = PPO.load(model_path)
    env = JansenEnv(
        render_mode=None,
        xml_path=xml_path,
        prismatic_cmd_limit=prismatic_cmd_limit,
        k_positive_amplitude=k_positive_amplitude,
        f_positive_amplitude=f_positive_amplitude,
        h_positive_amplitude=h_positive_amplitude,
        k_negative_amplitude=k_negative_amplitude,
        f_negative_amplitude=f_negative_amplitude,
        h_negative_amplitude=h_negative_amplitude,
        k_slew_rate=slew_rate,
        reset_settle_time=reset_settle_time,
        train_prismatic_groups=train_prismatic_groups,
    )
    
    all_rewards = []
    for i in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        print(f"  Episode {i+1}: {episode_reward:.2f}")

    print("\n" + "!"*40)
    print(f"BEST REWARD:  {np.max(all_rewards):.2f}")
    print(f"WORST REWARD: {np.min(all_rewards):.2f}")
    print(f"AVERAGE:      {np.mean(all_rewards):.2f}")
    print("!"*40)
    env.close()

def evaluate(
    model_path="jansen_rl_expert_2M",
    xml_path="jansen_assembly_red_articulated.xml",
    prismatic_cmd_limit=0.005,
    k_positive_amplitude=0.01,
    f_positive_amplitude=0.002,
    h_positive_amplitude=0.01,
    k_negative_amplitude=0.0,
    f_negative_amplitude=0.0,
    h_negative_amplitude=0.01,
    slew_rate=0.05,
    reset_settle_time=6.2,
    video_path=None,
    video_duration=10.0,
    video_fps=60,
    video_width=640,
    video_height=480,
    video_cam_azimuth=180.0,
    video_cam_elevation=-14.0,
    video_cam_distance=0.70,
    video_cam_lookat_y=0.0,
    video_cam_lookat_z=0.10,
    train_prismatic_groups="kfh",
):
    """
    Visual evaluation.
    """
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        return

    model = PPO.load(model_path)
    render_mode = None if video_path else "human"
    env = JansenEnv(
        render_mode=render_mode,
        xml_path=xml_path,
        prismatic_cmd_limit=prismatic_cmd_limit,
        k_positive_amplitude=k_positive_amplitude,
        f_positive_amplitude=f_positive_amplitude,
        h_positive_amplitude=h_positive_amplitude,
        k_negative_amplitude=k_negative_amplitude,
        f_negative_amplitude=f_negative_amplitude,
        h_negative_amplitude=h_negative_amplitude,
        k_slew_rate=slew_rate,
        reset_settle_time=reset_settle_time,
        train_prismatic_groups=train_prismatic_groups,
    )

    if video_path:
        video_dir = os.path.dirname(video_path) or "."
        os.makedirs(video_dir, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=video_height, width=video_width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = video_cam_azimuth
        cam.elevation = video_cam_elevation
        cam.distance = video_cam_distance
        cam.lookat[:] = [0.0, video_cam_lookat_y, video_cam_lookat_z]

        obs, _ = env.reset()
        simulated_time = 0.0
        total_frames = int(round(video_duration * video_fps))
        frames_written = 0

        with imageio.get_writer(video_path, fps=video_fps, codec="libx264", quality=8) as writer:
            for frame_idx in range(total_frames):
                target_time = frame_idx / float(video_fps)
                while simulated_time < target_time:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    simulated_time += env.dt
                    if terminated or truncated:
                        obs, _ = env.reset()

                cam.lookat[:] = [
                    float(env.data.qpos[env.plate_qadr]),
                    video_cam_lookat_y,
                    video_cam_lookat_z,
                ]
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
                frames_written += 1

        renderer.close()
        env.close()
        print(
            f"Eval video saved: {video_path} | fps={video_fps} | "
            f"duration={video_duration:.3f}s | frames={frames_written}"
        )
        return
    
    obs, _ = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jansen Walker RL: Parallel Training")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "analyze"], default="train")
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--model", type=str, default="jansen_rl_expert_2M")
    parser.add_argument("--load", type=str, default=None, help="Checkpoint model name/path (without .zip) to resume from")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gpu-heavy", action="store_true")
    parser.add_argument("--xml", type=str, default="jansen_assembly_red_articulated.xml")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--prismatic-cmd-limit", type=float, default=0.005, help="Prismatic target command magnitude limit in meters.")
    parser.add_argument("--k-positive-amplitude", type=float, default=0.01, help="K-link positive target limit in meters for all legs.")
    parser.add_argument("--f-positive-amplitude", type=float, default=0.002, help="F-link positive target limit in meters for all legs.")
    parser.add_argument("--h-positive-amplitude", type=float, default=0.01, help="H-link positive target limit in meters for all legs.")
    parser.add_argument("--k-negative-amplitude", type=float, default=0.0, help="K-link negative target magnitude in meters for all legs.")
    parser.add_argument("--f-negative-amplitude", type=float, default=0.0, help="F-link negative target magnitude in meters for all legs.")
    parser.add_argument("--h-negative-amplitude", type=float, default=0.01, help="H-link negative target magnitude in meters for all legs.")
    parser.add_argument("--slew-rate", type=float, default=0.05, help="Prismatic command slew rate in m/s.")
    parser.add_argument("--reset-settle-time", type=float, default=6.2, help="Reset settle duration in seconds.")
    parser.add_argument("--train-prismatic-groups", type=str, default="kfh", help="Which prismatic groups are policy-controlled: k, f, h, or combinations like kfh/h.")
    parser.add_argument("--video", type=str, default=None, help="Record eval video to this MP4 path.")
    parser.add_argument("--video-duration", type=float, default=10.0, help="Eval video duration in simulated seconds.")
    parser.add_argument("--video-fps", type=int, default=60, help="Eval video FPS.")
    parser.add_argument("--video-width", type=int, default=640, help="Eval video width.")
    parser.add_argument("--video-height", type=int, default=480, help="Eval video height.")
    parser.add_argument("--video-cam-azimuth", type=float, default=180.0, help="Eval video camera azimuth in degrees.")
    parser.add_argument("--video-cam-elevation", type=float, default=-14.0, help="Eval video camera elevation in degrees.")
    parser.add_argument("--video-cam-distance", type=float, default=0.70, help="Eval video camera distance.")
    parser.add_argument("--video-cam-lookat-y", type=float, default=0.0, help="Eval video camera lookat Y.")
    parser.add_argument("--video-cam-lookat-z", type=float, default=0.10, help="Eval video camera lookat Z.")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            timesteps=args.steps,
            model_name=args.model,
            load_model=args.load,
            n_envs=args.envs,
            render=args.render,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gpu_heavy=args.gpu_heavy,
            xml_path=args.xml,
            device_preference=args.device,
            prismatic_cmd_limit=args.prismatic_cmd_limit,
            k_positive_amplitude=args.k_positive_amplitude,
            f_positive_amplitude=args.f_positive_amplitude,
            h_positive_amplitude=args.h_positive_amplitude,
            k_negative_amplitude=args.k_negative_amplitude,
            f_negative_amplitude=args.f_negative_amplitude,
            h_negative_amplitude=args.h_negative_amplitude,
            slew_rate=args.slew_rate,
            reset_settle_time=args.reset_settle_time,
            train_prismatic_groups=args.train_prismatic_groups,
        )
    elif args.mode == "eval":
        evaluate(
            args.model,
            xml_path=args.xml,
            prismatic_cmd_limit=args.prismatic_cmd_limit,
            k_positive_amplitude=args.k_positive_amplitude,
            f_positive_amplitude=args.f_positive_amplitude,
            h_positive_amplitude=args.h_positive_amplitude,
            k_negative_amplitude=args.k_negative_amplitude,
            f_negative_amplitude=args.f_negative_amplitude,
            h_negative_amplitude=args.h_negative_amplitude,
            slew_rate=args.slew_rate,
            reset_settle_time=args.reset_settle_time,
            video_path=args.video,
            video_duration=args.video_duration,
            video_fps=args.video_fps,
            video_width=args.video_width,
            video_height=args.video_height,
            video_cam_azimuth=args.video_cam_azimuth,
            video_cam_elevation=args.video_cam_elevation,
            video_cam_distance=args.video_cam_distance,
            video_cam_lookat_y=args.video_cam_lookat_y,
            video_cam_lookat_z=args.video_cam_lookat_z,
            train_prismatic_groups=args.train_prismatic_groups,
        )
    elif args.mode == "analyze":
        analyze(
            args.model,
            xml_path=args.xml,
            prismatic_cmd_limit=args.prismatic_cmd_limit,
            k_positive_amplitude=args.k_positive_amplitude,
            f_positive_amplitude=args.f_positive_amplitude,
            h_positive_amplitude=args.h_positive_amplitude,
            k_negative_amplitude=args.k_negative_amplitude,
            f_negative_amplitude=args.f_negative_amplitude,
            h_negative_amplitude=args.h_negative_amplitude,
            slew_rate=args.slew_rate,
            reset_settle_time=args.reset_settle_time,
            train_prismatic_groups=args.train_prismatic_groups,
        )
