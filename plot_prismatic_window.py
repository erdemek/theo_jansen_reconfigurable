import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from jansen_world import JansenEnv


def record_prismatic(model_path):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file '{model_path}.zip' not found.")

    model = PPO.load(model_path)
    env = JansenEnv(render_mode=None)
    obs, _ = env.reset()

    times = []
    k_hist = []
    f_hist = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        times.append(env.data.time)
        k_hist.append(env.k_extension_states.copy())
        f_hist.append(env.f_extension_states.copy())

    env.close()
    return np.asarray(times), np.asarray(k_hist), np.asarray(f_hist)


def plot_window(times, values, name, out_dir, t_start, t_end):
    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        raise ValueError(f"No samples in window {t_start}-{t_end}s. Max time: {times[-1]:.3f}s")

    t = times[mask]
    v = values[mask]

    plt.figure(figsize=(10, 4))
    plt.plot(t, v, linewidth=1.5)
    plt.title(f"{name} extension ({t_start}-{t_end}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Extension (m)")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f"{name}_ext_{t_start:.0f}_{t_end:.0f}s.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_plots(model_path, out_dir, t_start=15.0, t_end=20.0):
    os.makedirs(out_dir, exist_ok=True)
    times, k_hist, f_hist = record_prismatic(model_path)

    for i in range(4):
        plot_window(times, k_hist[:, i], f"k{i+1}", out_dir, t_start, t_end)
    for i in range(4):
        plot_window(times, f_hist[:, i], f"f{i+1}", out_dir, t_start, t_end)


if __name__ == "__main__":
    model = "jansen_rl_kf_30M_roll_v1"
    out_dir = f"prismatic_plots_{model}_15_20s"
    generate_plots(model, out_dir, 15.0, 20.0)
    print(f"Saved plots to {out_dir}")
