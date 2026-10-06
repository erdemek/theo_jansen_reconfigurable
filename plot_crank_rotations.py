import argparse
import math
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

from jansen_world import JansenEnv
from train_jansen import resolve_model_path


CRANK_JOINTS = ["hinge_m", "hinge_m_r", "hinge_m_y", "hinge_m_r_y"]
CRANK_LABELS = ["front_left", "rear_left", "front_right", "rear_right"]


def joint_addresses(model, joint_names):
    qpos_adrs = []
    qvel_adrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing crank joint: {name}")
        qpos_adrs.append(model.jnt_qposadr[jid])
        qvel_adrs.append(model.jnt_dofadr[jid])
    return np.array(qpos_adrs, dtype=np.int32), np.array(qvel_adrs, dtype=np.int32)


def save_line_plot(path, title, xlabel, ylabel, x, series, labels, target=None):
    plt.figure(figsize=(11, 5.5))
    for values, label in zip(series.T, labels):
        plt.plot(x, values, linewidth=1.5, label=label)
    if target is not None:
        plt.plot(x, target, color="#111827", linestyle="--", linewidth=1.4, label="target")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_crank_rotations(
    model_path,
    xml_path,
    output_dir,
    duration,
    device,
    prismatic_cmd_limit,
    h_positive_amplitude,
    h_negative_amplitude,
    slew_rate,
    reset_settle_time,
    train_prismatic_groups,
    reward_preset,
):
    resolved_model = resolve_model_path(model_path)
    model = PPO.load(resolved_model, device=device)
    env = JansenEnv(
        render_mode=None,
        xml_path=xml_path,
        prismatic_cmd_limit=prismatic_cmd_limit,
        h_positive_amplitude=h_positive_amplitude,
        h_negative_amplitude=h_negative_amplitude,
        k_slew_rate=slew_rate,
        reset_settle_time=reset_settle_time,
        train_prismatic_groups=train_prismatic_groups,
        reward_preset=reward_preset,
    )

    qpos_adrs, qvel_adrs = joint_addresses(env.model, CRANK_JOINTS)
    obs, _ = env.reset()

    times = []
    qpos = []
    qvel = []
    rewards = []
    terminated = False
    truncated = False

    steps = int(math.ceil(duration / env.dt))
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        times.append(float(env.data.time))
        qpos.append(env.data.qpos[qpos_adrs].copy())
        qvel.append(env.data.qvel[qvel_adrs].copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    env.close()

    os.makedirs(output_dir, exist_ok=True)

    t = np.array(times, dtype=np.float64)
    raw_qpos = np.array(qpos, dtype=np.float64)
    angular_velocity = np.array(qvel, dtype=np.float64)
    unwrapped = np.unwrap(raw_qpos, axis=0)
    unwrapped_from_start = unwrapped - unwrapped[0:1]
    rotations = unwrapped_from_start / (2.0 * math.pi)
    wrapped_phase = np.mod(raw_qpos + math.pi, 2.0 * math.pi) - math.pi
    target_angle = env.speed_cmd * (t - t[0])
    target_rotations = target_angle / (2.0 * math.pi)

    save_line_plot(
        os.path.join(output_dir, "crank_rotation_angle_unwrapped.png"),
        "Crank Angle, Unwrapped",
        "Time (s)",
        "Angle since start (rad)",
        t - t[0],
        unwrapped_from_start,
        CRANK_LABELS,
        target=target_angle,
    )
    save_line_plot(
        os.path.join(output_dir, "crank_rotation_since_start_overlay.png"),
        "Crank Rotations Since Start",
        "Time (s)",
        "Rotations",
        t - t[0],
        rotations,
        CRANK_LABELS,
        target=target_rotations,
    )
    save_line_plot(
        os.path.join(output_dir, "crank_phase_wrapped.png"),
        "Crank Phase, Wrapped",
        "Time (s)",
        "Phase (rad)",
        t - t[0],
        wrapped_phase,
        CRANK_LABELS,
    )
    save_line_plot(
        os.path.join(output_dir, "crank_angular_velocity_overlay.png"),
        "Crank Angular Velocity",
        "Time (s)",
        "Angular velocity (rad/s)",
        t - t[0],
        angular_velocity,
        CRANK_LABELS,
        target=np.full_like(t, env.speed_cmd),
    )

    elapsed = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    mean_speed = (
        (unwrapped[-1] - unwrapped[0]) / elapsed
        if elapsed > 0.0
        else np.zeros(len(CRANK_JOINTS), dtype=np.float64)
    )
    print(f"Model: {resolved_model}")
    print(f"XML: {xml_path}")
    print(f"Saved: {output_dir}")
    print(f"Samples: {len(t)}")
    print(f"Sim duration: {elapsed:.4f}s")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Episode reward over window: {sum(rewards):.4f}")
    for label, rot, speed in zip(CRANK_LABELS, rotations[-1], mean_speed):
        print(f"{label}: rotations={rot:.4f}, mean_speed={speed:.4f} rad/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot crank rotations from a PPO checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prismatic-cmd-limit", type=float, default=0.005)
    parser.add_argument("--h-positive-amplitude", type=float, default=0.01)
    parser.add_argument("--h-negative-amplitude", type=float, default=0.01)
    parser.add_argument("--slew-rate", type=float, default=0.05)
    parser.add_argument("--reset-settle-time", type=float, default=6.2)
    parser.add_argument("--train-prismatic-groups", type=str, default="kfh")
    parser.add_argument("--reward-preset", type=str, default="h_only_straight_walk")
    args = parser.parse_args()

    plot_crank_rotations(
        model_path=args.model,
        xml_path=args.xml,
        output_dir=args.out_dir,
        duration=args.duration,
        device=args.device,
        prismatic_cmd_limit=args.prismatic_cmd_limit,
        h_positive_amplitude=args.h_positive_amplitude,
        h_negative_amplitude=args.h_negative_amplitude,
        slew_rate=args.slew_rate,
        reset_settle_time=args.reset_settle_time,
        train_prismatic_groups=args.train_prismatic_groups,
        reward_preset=args.reward_preset,
    )
