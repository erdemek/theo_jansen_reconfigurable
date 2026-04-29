import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive slide_k with random piecewise-constant commands and record the response."
    )
    parser.add_argument("--xml", type=str, default="jansen_assembly.xml")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-hold", type=float, default=0.15, help="Minimum time before picking a new command.")
    parser.add_argument("--max-hold", type=float, default=0.9, help="Maximum time before picking a new command.")
    parser.add_argument("--cmd-min", type=float, default=-0.0035, help="Minimum commanded extension in meters.")
    parser.add_argument("--cmd-max", type=float, default=0.0035, help="Maximum commanded extension in meters.")
    parser.add_argument("--slew-rate", type=float, default=0.01, help="Command slew rate in m/s.")
    parser.add_argument("--actuator-clamp", type=float, default=0.005, help="Clamp applied before writing control.")
    parser.add_argument("--speed-cmd", type=float, default=5.0, help="Crank velocity command.")
    parser.add_argument("--target-joint", type=str, default="slide_k")
    parser.add_argument("--target-actuator", type=str, default="k1_prismatic")
    parser.add_argument("--out-plot", type=str, default="plots/random_prismatic_commands.png")
    parser.add_argument("--out-csv", type=str, default="plots/random_prismatic_commands.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    crank_names = ["crank_motor", "crank_motor_r", "crank_motor_y", "crank_motor_r_y"]
    crank_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in crank_names]
    if min(crank_ids) < 0:
        raise RuntimeError("One or more crank actuators were not found in the XML.")

    target_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.target_actuator)
    if target_actuator_id < 0:
        raise RuntimeError(f"Actuator '{args.target_actuator}' was not found in the XML.")

    target_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, args.target_joint)
    if target_joint_id < 0:
        raise RuntimeError(f"Joint '{args.target_joint}' was not found in the XML.")
    target_qadr = model.jnt_qposadr[target_joint_id]

    dt = model.opt.timestep
    steps = int(round(args.duration / dt))
    if steps <= 0:
        raise RuntimeError("Duration is too short for this model timestep.")

    current_cmd = 0.0
    target_cmd = rng.uniform(args.cmd_min, args.cmd_max)
    next_switch_time = rng.uniform(args.min_hold, args.max_hold)

    times = []
    qpos_values = []
    ctrl_values = []
    target_values = []

    for _ in range(steps):
        if data.time >= next_switch_time:
            target_cmd = rng.uniform(args.cmd_min, args.cmd_max)
            next_switch_time = data.time + rng.uniform(args.min_hold, args.max_hold)

        max_delta = args.slew_rate * dt
        current_cmd += clamp(target_cmd - current_cmd, -max_delta, max_delta)

        for crank_id in crank_ids:
            data.ctrl[crank_id] = args.speed_cmd
        data.ctrl[target_actuator_id] = clamp(current_cmd, -args.actuator_clamp, args.actuator_clamp)

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        times.append(data.time)
        qpos_values.append(float(data.qpos[target_qadr]))
        ctrl_values.append(float(data.ctrl[target_actuator_id]))
        target_values.append(float(target_cmd))

    times = np.asarray(times)
    qpos_values = np.asarray(qpos_values)
    ctrl_values = np.asarray(ctrl_values)
    target_values = np.asarray(target_values)

    plot_dir = os.path.dirname(args.out_plot)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    csv_dir = os.path.dirname(args.out_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "joint_qpos_m", "ctrl_m", "target_m"])
        writer.writerows(zip(times, qpos_values, ctrl_values, target_values))

    plt.figure(figsize=(11, 5.5))
    plt.plot(times, qpos_values, label=f"{args.target_joint} qpos", linewidth=1.7)
    plt.plot(times, ctrl_values, label=f"{args.target_actuator} ctrl", linewidth=1.1, alpha=0.9)
    plt.plot(times, target_values, label="random target", linewidth=1.0, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Extension (m)")
    plt.title("Random prismatic command tracking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=170)
    plt.close()

    print(f"Saved plot: {args.out_plot}")
    print(f"Saved csv: {args.out_csv}")
    print(
        f"Loaded settings: speed_cmd={args.speed_cmd}, "
        f"slew_rate={args.slew_rate}, cmd_range=[{args.cmd_min}, {args.cmd_max}]"
    )
    print(f"{args.target_joint} min/max: {qpos_values.min():.6f} / {qpos_values.max():.6f} m")
    print(f"{args.target_actuator} ctrl min/max: {ctrl_values.min():.6f} / {ctrl_values.max():.6f} m")


if __name__ == "__main__":
    main()
