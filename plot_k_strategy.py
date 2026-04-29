import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from jansen_world import JansenEnv


def record_k_command_vs_output(model_path):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file '{model_path}.zip' not found.")

    model = PPO.load(model_path)
    env = JansenEnv(render_mode=None)
    obs, _ = env.reset()

    times = []
    target_cmd_hist = []
    slew_cmd_hist = []
    output_hist = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        # Same command scaling as in JansenEnv.step for k-link targets.
        target_k = np.clip(action, -1.0, 1.0)[0:4] * env.prismatic_cmd_limit

        obs, reward, terminated, truncated, _ = env.step(action)

        times.append(env.data.time)
        target_cmd_hist.append(target_k.copy())
        slew_cmd_hist.append(env.k_extension_states.copy())
        output_hist.append(np.array([env.data.qpos[qadr] for qadr in env.slide_k_qadrs], dtype=np.float64))

    env.close()
    return (
        np.asarray(times),
        np.asarray(target_cmd_hist),
        np.asarray(slew_cmd_hist),
        np.asarray(output_hist),
    )


def plot_k_command_vs_output(model_path, output_file="k_command_vs_output.png"):
    print(f"Recording command vs output for model: {model_path}")
    times, target_cmd, slew_cmd, outputs = record_k_command_vs_output(model_path)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    for i in range(4):
        ax = axes[i]
        ax.plot(times, target_cmd[:, i], color="tab:orange", alpha=0.8, linewidth=1.2, label=f"K{i+1} target cmd")
        ax.plot(times, slew_cmd[:, i], color="tab:blue", alpha=0.9, linewidth=1.5, label=f"K{i+1} actuator cmd")
        ax.plot(times, outputs[:, i], color="tab:green", alpha=0.9, linewidth=1.5, label=f"K{i+1} joint output")
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"K-Link Command vs Output\nModel: {model_path} | Duration: {times[-1]:.2f}s")
    fig.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot k-link command vs output for a trained model")
    parser.add_argument("--model", type=str, default="jansen_rl_kf_30M_roll_v1")
    parser.add_argument("--out", type=str, default="k_command_vs_output.png")
    args = parser.parse_args()

    plot_k_command_vs_output(args.model, args.out)
