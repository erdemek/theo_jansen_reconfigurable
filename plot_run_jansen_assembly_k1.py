import argparse
import ast
import math
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_run_jansen_assembly_constants(script_path: str) -> dict:
    with open(script_path, "r", encoding="utf-8-sig") as f:
        tree = ast.parse(f.read(), filename=script_path)

    wanted = {
        "speed_cmd",
        "initial_body_vx",
        "k1_extension_cmd",
        "k1_slew_rate",
        "settle_damping",
        "run_damping",
        "target_delta",
        "position_kp",
        "position_max_cmd",
        "position_tol",
        "velocity_tol",
    }
    values = {}
    safe_globals = {"__builtins__": {}, "math": math}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in wanted:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
        except Exception:
            try:
                expr = ast.Expression(node.value)
                compiled = compile(expr, script_path, "eval")
                values[name] = eval(compiled, safe_globals, {})
            except Exception:
                continue

    missing = sorted(wanted - values.keys())
    if missing:
        raise RuntimeError(
            "Could not read these constants from run_jansen_assembly.py: "
            + ", ".join(missing)
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot slide_k extension over time using run_jansen_assembly logic.")
    parser.add_argument("--xml", type=str, default="jansen_assembly.xml")
    parser.add_argument("--script", type=str, default="run_jansen_assembly.py")
    parser.add_argument("--out", type=str, default="plots/run_jansen_assembly_k1_extension.png")
    parser.add_argument("--csv", type=str, default="plots/run_jansen_assembly_k1_extension.csv")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--idle-settle-time", type=float, default=0.2)
    parser.add_argument("--position-settle-time", type=float, default=1.0)
    parser.add_argument("--lock-air", action="store_true")
    parser.add_argument("--lock-air-z", type=float, default=0.22)
    args = parser.parse_args()

    script_values = load_run_jansen_assembly_constants(args.script)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")
    act_id_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_r")
    act_id_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_y")
    act_id_r_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_r_y")
    if min(act_id, act_id_r, act_id_y, act_id_r_y) < 0:
        raise RuntimeError("One or more crank actuators were not found in jansen_assembly.xml")

    k1_prismatic_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "k1_prismatic")
    k2_prismatic_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "k2_prismatic")
    k3_prismatic_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "k3_prismatic")
    k4_prismatic_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "k4_prismatic")
    if min(k1_prismatic_act_id, k2_prismatic_act_id, k3_prismatic_act_id, k4_prismatic_act_id) < 0:
        raise RuntimeError("One or more k-link actuators were not found in jansen_assembly.xml")

    slide_k_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_k")
    if slide_k_jid < 0:
        raise RuntimeError("Joint 'slide_k' was not found in jansen_assembly.xml")
    slide_k_qadr = model.jnt_qposadr[slide_k_jid]

    jid_m = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_m")
    jid_m_r_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_m_r_y")
    if min(jid_m, jid_m_r_y) < 0:
        raise RuntimeError("One or more crank joints were not found in jansen_assembly.xml")
    qadr_m = model.jnt_qposadr[jid_m]
    qadr_m_r_y = model.jnt_qposadr[jid_m_r_y]
    dof_m = model.jnt_dofadr[jid_m]
    dof_m_r_y = model.jnt_dofadr[jid_m_r_y]

    plate_free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    if plate_free_jid < 0:
        raise RuntimeError("Joint 'plate_free' not found. Check jansen_assembly.xml")
    plate_qadr = model.jnt_qposadr[plate_free_jid]
    plate_vadr = model.jnt_dofadr[plate_free_jid]
    plate_qpos0 = data.qpos[plate_qadr:plate_qadr + 7].copy()
    if args.lock_air:
        plate_qpos0[2] = float(args.lock_air_z)

    settle_damping = float(script_values["settle_damping"])
    run_damping = float(script_values["run_damping"])
    hinge_dof_ids = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_dof_ids.append(model.jnt_dofadr[j])

    def set_hinge_damping(value: float) -> None:
        for dof_id in hinge_dof_ids:
            model.dof_damping[dof_id] = value

    def lock_plate_to_initial_pose() -> None:
        data.qpos[plate_qadr:plate_qadr + 7] = plate_qpos0
        data.qvel[plate_vadr:plate_vadr + 6] = 0.0

    def zero_all_cranks() -> None:
        data.ctrl[act_id] = 0.0
        data.ctrl[act_id_r] = 0.0
        data.ctrl[act_id_y] = 0.0
        data.ctrl[act_id_r_y] = 0.0

    def set_k_extension(ext_cmd: float) -> None:
        val = clamp(ext_cmd, -0.005, 0.005)
        data.ctrl[k1_prismatic_act_id] = val
        data.ctrl[k2_prismatic_act_id] = val
        data.ctrl[k3_prismatic_act_id] = val
        data.ctrl[k4_prismatic_act_id] = val

    speed_cmd = float(script_values["speed_cmd"])
    initial_body_vx = float(script_values["initial_body_vx"])
    k1_extension_cmd = float(script_values["k1_extension_cmd"])
    k1_extension_state = 0.0
    k1_slew_rate = float(script_values["k1_slew_rate"])

    idle_settle_time = args.idle_settle_time
    position_settle_time = args.position_settle_time
    min_settle_end_time = idle_settle_time + position_settle_time

    target_delta = float(script_values["target_delta"])
    position_kp = float(script_values["position_kp"])
    position_max_cmd = float(script_values["position_max_cmd"])
    position_tol = float(script_values["position_tol"])
    velocity_tol = float(script_values["velocity_tol"])

    stagger_act_1 = act_id
    stagger_act_2 = act_id_r_y
    target_q_m = float(data.qpos[qadr_m] + target_delta)
    target_q_m_r_y = float(data.qpos[qadr_m_r_y] + target_delta)

    set_hinge_damping(settle_damping)
    damping_switched = False
    position_targets_reached = False
    initial_velocity_applied = False
    if args.lock_air:
        position_targets_reached = True

    k1_extension_state = float(data.qpos[slide_k_qadr])
    run_phase_started = False
    crank_angle_at_run_start = 0.0
    k1_triggered = False

    times = []
    slide_k_values = []
    ctrl_values = []

    for _ in range(args.steps):
        sim_time = data.time
        idle_phase = sim_time < idle_settle_time
        position_settle_phase = (not idle_phase) and (not position_targets_reached)

        if idle_phase:
            lock_plate_to_initial_pose()
            zero_all_cranks()
            max_delta = k1_slew_rate * model.opt.timestep
            k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
            set_k_extension(k1_extension_state)
        elif position_settle_phase:
            lock_plate_to_initial_pose()
            zero_all_cranks()
            max_delta = k1_slew_rate * model.opt.timestep
            k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
            set_k_extension(k1_extension_state)

            err_m = target_q_m - float(data.qpos[qadr_m])
            err_m_r_y = target_q_m_r_y - float(data.qpos[qadr_m_r_y])
            data.ctrl[stagger_act_1] = clamp(position_kp * err_m, -position_max_cmd, position_max_cmd)
            data.ctrl[stagger_act_2] = clamp(position_kp * err_m_r_y, -position_max_cmd, position_max_cmd)

            reached_m = abs(err_m) <= position_tol and abs(float(data.qvel[dof_m])) <= velocity_tol
            reached_m_r_y = abs(err_m_r_y) <= position_tol and abs(float(data.qvel[dof_m_r_y])) <= velocity_tol
            if sim_time >= min_settle_end_time and reached_m and reached_m_r_y:
                position_targets_reached = True
        else:
            lock_plate_to_initial_pose()
            if not damping_switched:
                set_hinge_damping(run_damping)
                damping_switched = True
            if not initial_velocity_applied:
                data.qvel[plate_vadr] = initial_body_vx
                initial_velocity_applied = True
            if not run_phase_started:
                crank_angle_at_run_start = float(data.qpos[qadr_m])
                run_phase_started = True

            crank_traveled = abs(float(data.qpos[qadr_m]) - crank_angle_at_run_start)
            if not k1_triggered and crank_traveled >= 0.1 * math.pi:
                k1_extension_cmd = 0.0035
                k1_triggered = True

            if k1_triggered:
                k1_tol = 0.0001
                k1_actual = float(data.qpos[slide_k_qadr])
                if abs(k1_actual - k1_extension_cmd) <= k1_tol:
                    k1_extension_cmd = -k1_extension_cmd

            data.ctrl[act_id] = speed_cmd
            data.ctrl[act_id_r] = speed_cmd
            data.ctrl[act_id_y] = speed_cmd
            data.ctrl[act_id_r_y] = speed_cmd

            max_delta = k1_slew_rate * model.opt.timestep
            k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
            set_k_extension(k1_extension_state)

        mujoco.mj_step(model, data)
        if args.lock_air or idle_phase or position_settle_phase:
            lock_plate_to_initial_pose()
        mujoco.mj_forward(model, data)

        times.append(data.time)
        slide_k_values.append(float(data.qpos[slide_k_qadr]))
        ctrl_values.append(float(data.ctrl[k1_prismatic_act_id]))

    times = np.asarray(times)
    slide_k_values = np.asarray(slide_k_values)
    ctrl_values = np.asarray(ctrl_values)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    csv_dir = os.path.dirname(args.csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    plt.figure(figsize=(11, 5.5))
    plt.plot(times, slide_k_values, label="slide_k qpos", linewidth=1.8)
    plt.plot(times, ctrl_values, label="k1_prismatic ctrl", linewidth=1.2, alpha=0.85)
    plt.xlabel("Time (s)")
    plt.ylabel("Extension (m)")
    plt.title("run_jansen_assembly: slide_k extension over time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    plt.close()

    np.savetxt(
        args.csv,
        np.column_stack([times, slide_k_values, ctrl_values]),
        delimiter=",",
        header="time_s,slide_k_qpos_m,k1_prismatic_ctrl_m",
        comments="",
    )

    print(f"Saved plot: {args.out}")
    print(f"Saved csv: {args.csv}")
    print(f"Source script: {args.script}")
    print(
        "Loaded values: "
        f"speed_cmd={speed_cmd}, "
        f"k1_extension_cmd={k1_extension_cmd}, "
        f"k1_slew_rate={k1_slew_rate}"
    )
    print(f"Final time: {times[-1]:.6f} s")
    print(f"slide_k min/max: {slide_k_values.min():.6f} / {slide_k_values.max():.6f} m")
    print(f"ctrl min/max: {ctrl_values.min():.6f} / {ctrl_values.max():.6f} m")


if __name__ == "__main__":
    main()
