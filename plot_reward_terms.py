import argparse
import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_ppo_dir(base_dir):
    runs = [
        d for d in glob.glob(os.path.join(base_dir, "PPO_*"))
        if os.path.isdir(d)
    ]
    if not runs:
        raise RuntimeError(f"No PPO run directory found under: {base_dir}")
    return max(runs, key=os.path.getmtime)


def collect_scalars(event_accumulator, tag, min_step=0):
    values = [value for value in event_accumulator.Scalars(tag) if value.step >= min_step]
    return [v.step for v in values], [v.value for v in values]


def plot_reward_terms(run_dir, output_file, min_step=0):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    term_tags = sorted(t for t in tags if t.startswith("rollout_reward_terms/"))
    if not term_tags:
        raise RuntimeError(f"No rollout_reward_terms scalars found in: {run_dir}")

    total_steps, total_values = collect_scalars(ea, "rollout/ep_rew_mean", min_step)
    terms = {
        tag.replace("rollout_reward_terms/", "").replace("_mean", ""):
        collect_scalars(ea, tag, min_step)
        for tag in term_tags
    }
    if not total_values:
        raise RuntimeError(f"No reward samples at or after step {min_step}: {run_dir}")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    colors = {
        "forward_progress": "#2563eb",
        "forward_speed": "#16a34a",
        "survival": "#ca8a04",
        "fall": "#dc2626",
        "roll_pitch_rate": "#7c3aed",
        "side_roll": "#db2777",
        "action_magnitude": "#ea580c",
    }
    positives = ["forward_progress", "forward_speed", "survival"]
    penalties = ["fall", "roll_pitch_rate", "side_roll", "action_magnitude"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(total_steps, total_values, color="#111827", linewidth=1.8, label="total")
    axes[0].set_ylabel("total reward")
    axes[0].legend(loc="best")

    for name in positives:
        if name in terms:
            steps, values = terms[name]
            axes[1].plot(steps, values, linewidth=1.5, label=name, color=colors.get(name))
    axes[1].set_ylabel("positive terms")
    axes[1].legend(loc="best")

    for name in penalties:
        if name in terms:
            steps, values = terms[name]
            axes[2].plot(steps, values, linewidth=1.5, label=name, color=colors.get(name))
    axes[2].set_ylabel("penalty terms")
    axes[2].set_xlabel("timesteps")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(alpha=0.28)

    fig.suptitle(f"Reward Term Decomposition ({os.path.basename(run_dir)})")
    fig.tight_layout()
    fig.savefig(output_file, dpi=170)
    plt.close(fig)

    print(f"Run: {run_dir}")
    print(f"Saved: {output_file}")
    print(f"Last total: {total_values[-1]:.4f}")
    for name, (_, values) in terms.items():
        print(
            f"{name}: last={values[-1]:.4f}, "
            f"best={max(values):.4f}, worst={min(values):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO reward-term decomposition.")
    parser.add_argument("--tb-root", type=str, default="tensorboard_logs")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default="plots/reward_terms_decomposition.png")
    parser.add_argument("--min-step", type=int, default=0, help="Plot samples at or after this timestep.")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_ppo_dir(args.tb_root)
    plot_reward_terms(run_dir, args.out, min_step=args.min_step)
