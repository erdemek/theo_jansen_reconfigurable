import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from jansen_world import JansenEnv


PRISMATIC_GROUPS = {
    "k": {
        "names": ["k1", "k2", "k3", "k4"],
        "act_attr": "k_act_ids",
        "qadr_attr": "slide_k_qadrs",
        "state_attr": "k_extension_states",
    },
    "f": {
        "names": ["f1", "f2", "f3", "f4"],
        "act_attr": "f_act_ids",
        "qadr_attr": "slide_f_qadrs",
        "state_attr": "f_extension_states",
    },
    "h": {
        "names": ["h1", "h2", "h3", "h4"],
        "act_attr": "h_act_ids",
        "qadr_attr": "slide_h_qadrs",
        "state_attr": "h_extension_states",
    },
}


def require_model_zip(model_path):
    if os.path.exists(model_path):
        return model_path
    zip_path = model_path if model_path.lower().endswith(".zip") else model_path + ".zip"
    if os.path.exists(zip_path):
        return zip_path
    raise FileNotFoundError(f"Model file not found. Tried: {model_path} and {zip_path}")


def record_command_response(args):
    model_path = require_model_zip(args.model)
    model = PPO.load(model_path, device=args.device)
    env = JansenEnv(
        render_mode=None,
        xml_path=args.xml,
        k_positive_amplitude=args.k_positive_amplitude,
        f_positive_amplitude=args.f_positive_amplitude,
        h_positive_amplitude=args.h_positive_amplitude,
        k_negative_amplitude=args.k_negative_amplitude,
        f_negative_amplitude=args.f_negative_amplitude,
        h_negative_amplitude=args.h_negative_amplitude,
        k_slew_rate=args.slew_rate,
        prismatic_max_acceleration=args.prismatic_max_acceleration,
        reset_settle_time=args.reset_settle_time,
        train_prismatic_groups=args.train_prismatic_groups,
        reward_preset=args.reward_preset,
    )
    obs, _ = env.reset()

    group_keys = [g for g in "kfh" if g in args.groups]
    times = []
    commanded = {g: [] for g in group_keys}
    response = {g: [] for g in group_keys}
    rewards = []

    terminated = False
    truncated = False
    max_steps = int(round(args.duration / env.dt)) if args.duration > 0.0 else env.max_steps
    step_count = 0

    while not (terminated or truncated) and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, _ = env.step(action)

        times.append(float(env.data.time))
        rewards.append(float(reward))
        for group in group_keys:
            meta = PRISMATIC_GROUPS[group]
            act_ids = getattr(env, meta["act_attr"])
            qadrs = getattr(env, meta["qadr_attr"])
            commanded[group].append(np.array([env.data.ctrl[aid] for aid in act_ids], dtype=np.float64))
            response[group].append(np.array([env.data.qpos[qadr] for qadr in qadrs], dtype=np.float64))

        step_count += 1

    env.close()

    return (
        np.asarray(times, dtype=np.float64),
        {g: np.asarray(values, dtype=np.float64) for g, values in commanded.items()},
        {g: np.asarray(values, dtype=np.float64) for g, values in response.items()},
        np.asarray(rewards, dtype=np.float64),
        terminated,
        truncated,
    )


def plot_group(times, commanded, response, group, out_dir, title_suffix):
    names = PRISMATIC_GROUPS[group]["names"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        ax.plot(times, commanded[:, i], color=color, linewidth=1.5, label=f"{names[i]} command")
        ax.plot(times, response[:, i], color=color, linewidth=1.2, linestyle="--", label=f"{names[i]} response")
        ax.set_title(names[i], loc="left", fontsize=10)
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{group.upper()} prismatic command vs response{title_suffix}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{group}_command_response.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_overlay(times, commanded, response, group, out_dir, title_suffix):
    names = PRISMATIC_GROUPS[group]["names"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        ax.plot(
            times,
            commanded[:, i],
            color=color,
            linewidth=1.5,
            label=f"{name} command",
        )
        ax.plot(
            times,
            response[:, i],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"{name} response",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("extension (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"{group.upper()} prismatic command vs response by color{title_suffix}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{group}_command_response_overlay.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_rates(times, commanded, response, group, out_dir, title_suffix):
    names = PRISMATIC_GROUPS[group]["names"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    if len(times) < 2:
        raise RuntimeError("Need at least two samples to plot change rates.")

    rate_times = times[1:]
    dt = np.diff(times)
    commanded_rates = np.diff(commanded, axis=0) / dt[:, None]
    response_rates = np.diff(response, axis=0) / dt[:, None]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        ax.plot(
            rate_times,
            commanded_rates[:, i],
            color=color,
            linewidth=1.5,
            label=f"{names[i]} command rate",
        )
        ax.plot(
            rate_times,
            response_rates[:, i],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"{names[i]} response rate",
        )
        ax.axhline(0.0, color="#666666", linewidth=0.7, alpha=0.45)
        ax.set_title(names[i], loc="left", fontsize=10)
        ax.set_ylabel("m/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{group.upper()} prismatic change rates{title_suffix}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{group}_change_rate.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_accelerations(times, commanded, response, group, out_dir, title_suffix):
    names = PRISMATIC_GROUPS[group]["names"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    if len(times) < 3:
        raise RuntimeError("Need at least three samples to plot accelerations.")

    dt = np.diff(times)
    commanded_rates = np.diff(commanded, axis=0) / dt[:, None]
    response_rates = np.diff(response, axis=0) / dt[:, None]
    rate_times = 0.5 * (times[1:] + times[:-1])
    acceleration_dt = np.diff(rate_times)
    acceleration_times = 0.5 * (rate_times[1:] + rate_times[:-1])
    commanded_accelerations = np.diff(commanded_rates, axis=0) / acceleration_dt[:, None]
    response_accelerations = np.diff(response_rates, axis=0) / acceleration_dt[:, None]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        ax.plot(
            acceleration_times,
            commanded_accelerations[:, i],
            color=color,
            linewidth=1.2,
            label=f"{names[i]} command acceleration",
        )
        ax.plot(
            acceleration_times,
            response_accelerations[:, i],
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=f"{names[i]} response acceleration",
        )
        ax.axhline(0.0, color="#666666", linewidth=0.7, alpha=0.45)
        ax.set_title(names[i], loc="left", fontsize=10)
        ax.set_ylabel("m/s²")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{group.upper()} prismatic acceleration{title_suffix}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{group}_acceleration.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(times, commanded, response, groups, out_dir, title_suffix):
    total_rows = 4 * len(groups)
    fig, axes = plt.subplots(total_rows, 1, figsize=(13, 2.15 * total_rows), sharex=True)
    if total_rows == 1:
        axes = [axes]

    row = 0
    for group in groups:
        names = PRISMATIC_GROUPS[group]["names"]
        for i in range(4):
            ax = axes[row]
            ax.plot(times, commanded[group][:, i], color="tab:blue", linewidth=1.3, label="command")
            ax.plot(times, response[group][:, i], color="tab:orange", linewidth=1.1, label="response")
            ax.set_ylabel(f"{names[i]}\n(m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            row += 1

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Prismatic command vs response{title_suffix}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "all_prismatic_command_response.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot prismatic actuator command and joint response over time.")
    parser.add_argument("--model", required=True, help="Model path without .zip")
    parser.add_argument("--xml", default="jansen_assembly_rolling_hills.xml")
    parser.add_argument("--out-dir", default="plots/prismatic_command_response")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to record. Use 0 for full episode.")
    parser.add_argument("--groups", default="kfh", help="Prismatic groups to plot: any combination of k, f, h")
    parser.add_argument("--train-prismatic-groups", default="h")
    parser.add_argument("--reward-preset", default="h_only_straight_walk")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--k-positive-amplitude", type=float, default=0.01)
    parser.add_argument("--f-positive-amplitude", type=float, default=0.002)
    parser.add_argument("--h-positive-amplitude", type=float, default=0.02)
    parser.add_argument("--k-negative-amplitude", type=float, default=0.0)
    parser.add_argument("--f-negative-amplitude", type=float, default=0.0)
    parser.add_argument("--h-negative-amplitude", type=float, default=0.02)
    parser.add_argument("--slew-rate", type=float, default=0.05)
    parser.add_argument(
        "--prismatic-max-acceleration",
        type=float,
        default=0.0,
        help="Post-PPO29 trial acceleration limit in m/s^2; 0 disables it.",
    )
    parser.add_argument("--reset-settle-time", type=float, default=6.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    groups = [g for g in "kfh" if g in args.groups.lower()]
    if not groups:
        raise ValueError("--groups must include at least one of k, f, h")

    times, commanded, response, rewards, terminated, truncated = record_command_response(args)
    if len(times) == 0:
        raise RuntimeError("No samples recorded.")

    title_suffix = (
        f"\n{os.path.basename(args.model)} | duration={times[-1]:.2f}s | "
        f"terminated={terminated} | truncated={truncated}"
    )
    saved = [plot_all(times, commanded, response, groups, args.out_dir, title_suffix)]
    for group in groups:
        saved.append(plot_group(times, commanded[group], response[group], group, args.out_dir, title_suffix))
        saved.append(plot_group_overlay(times, commanded[group], response[group], group, args.out_dir, title_suffix))
        saved.append(plot_group_rates(times, commanded[group], response[group], group, args.out_dir, title_suffix))
        saved.append(plot_group_accelerations(times, commanded[group], response[group], group, args.out_dir, title_suffix))

    print(f"Samples: {len(times)}")
    print(f"Time: {times[0]:.4f}s to {times[-1]:.4f}s")
    print(f"Episode reward over recorded window: {float(np.sum(rewards)):.4f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
