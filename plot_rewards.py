import argparse
import glob
import os
from collections import deque

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_ppo_dir(base_dir: str) -> str:
    runs = [
        d for d in glob.glob(os.path.join(base_dir, "PPO_*"))
        if os.path.isdir(d)
    ]
    if not runs:
        raise RuntimeError(f"No PPO run directory found under: {base_dir}")
    return max(runs, key=os.path.getmtime)


def collect_reward_points(log_dir: str):
    event_files = sorted(
        glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
        key=os.path.getmtime,
    )
    if not event_files:
        raise RuntimeError(f"No TensorBoard event files found in: {log_dir}")

    points = []
    for event_file in event_files:
        ea = EventAccumulator(event_file)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "rollout/ep_rew_mean" not in tags:
            continue
        for e in ea.Scalars("rollout/ep_rew_mean"):
            points.append((e.step, e.value, e.wall_time))

    if not points:
        raise RuntimeError(
            "No 'rollout/ep_rew_mean' scalar found yet. "
            "Training may not have logged rewards yet."
        )

    # Keep newest wall-time entry for each step to merge resumed streams.
    by_step = {}
    for step, value, wall_time in points:
        if (step not in by_step) or (wall_time > by_step[step][1]):
            by_step[step] = (value, wall_time)

    steps = sorted(by_step.keys())
    values = [by_step[s][0] for s in steps]
    return steps, values


def moving_average(values, window):
    q = deque()
    total = 0.0
    out = []
    for v in values:
        q.append(v)
        total += v
        if len(q) > window:
            total -= q.popleft()
        out.append(total / len(q))
    return out


def plot_rewards(
    tensorboard_root: str,
    run_dir: str = None,
    output_file: str = "plots/reward_current_active.png",
    from_step: int = None,
    to_step: int = None,
):
    if run_dir is None:
        run_dir = latest_ppo_dir(tensorboard_root)

    steps, values = collect_reward_points(run_dir)

    filtered = [
        (s, v)
        for s, v in zip(steps, values)
        if (from_step is None or s >= from_step)
        and (to_step is None or s <= to_step)
    ]
    if not filtered:
        raise RuntimeError("No points in selected step range.")

    x = [p[0] for p in filtered]
    y = [p[1] for p in filtered]
    w = max(20, len(y) // 80)
    smooth = moving_average(y, w)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(11, 5.5))
    plt.plot(x, y, color="#1f77b4", linewidth=1.7, label="ep_rew_mean")
    plt.plot(x, smooth, color="#d62728", linewidth=2.0, label=f"moving avg (w={w})")
    plt.title(f"Reward Progress ({os.path.basename(run_dir)})")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=170)

    print(f"Run: {run_dir}")
    print(f"Saved: {output_file}")
    print(f"First: step={x[0]}, reward={y[0]:.4f}")
    print(f"Last: step={x[-1]}, reward={y[-1]:.4f}")
    print(f"Best: reward={max(y):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO reward curve from TensorBoard logs.")
    parser.add_argument("--tb-root", type=str, default="tensorboard_logs")
    parser.add_argument("--run-dir", type=str, default=None, help="Example: tensorboard_logs/PPO_28")
    parser.add_argument("--out", type=str, default="plots/reward_current_active.png")
    parser.add_argument("--from-step", type=int, default=None)
    parser.add_argument("--to-step", type=int, default=None)
    args = parser.parse_args()

    plot_rewards(
        tensorboard_root=args.tb_root,
        run_dir=args.run_dir,
        output_file=args.out,
        from_step=args.from_step,
        to_step=args.to_step,
    )
