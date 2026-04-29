import argparse
import math
import os
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_actuator_id(model: mujoco.MjModel, name: str, required: bool = True) -> int:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if required and act_id < 0:
        raise RuntimeError(f"Actuator '{name}' was not found in jansen_assembly.xml")
    return act_id


def get_joint_id(model: mujoco.MjModel, name: str, required: bool = True) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if required and jid < 0:
        raise RuntimeError(f"Joint '{name}' was not found in jansen_assembly.xml")
    return jid


parser = argparse.ArgumentParser(description="Run Jansen with grouped-phase prismatic-only gait")
parser.add_argument("--video", type=str, default=None, help="Output MP4 path")
parser.add_argument("--video-steps", type=int, default=40000)
parser.add_argument("--video-width", type=int, default=1920)
parser.add_argument("--video-height", type=int, default=1080)
parser.add_argument("--video-fps", type=int, default=60)
parser.add_argument("--video-frame-skip", type=int, default=0)
parser.add_argument("--cam-azimuth", type=float, default=None)
parser.add_argument("--cam-elevation", type=float, default=None)
parser.add_argument("--cam-distance", type=float, default=None)
parser.add_argument("--cam-lookat-x", type=float, default=0.0)
parser.add_argument("--cam-lookat-y", type=float, default=0.0)
parser.add_argument("--cam-lookat-z", type=float, default=0.12)
parser.add_argument("--idle-settle-time", type=float, default=0.2)
parser.add_argument("--position-settle-time", type=float, default=1.0)
parser.add_argument("--lock-air", action="store_true", help="Keep plate fixed while legs move")
parser.add_argument("--lock-air-z", type=float, default=0.22, help="Plate center Z when --lock-air is used")
parser.add_argument("--prismatic-amplitude", type=float, default=0.000, help="Extension amplitude in meters")
parser.add_argument("--prismatic-positive-amplitude", type=float, default=None, help="Positive prismatic peak in meters")
parser.add_argument("--prismatic-negative-amplitude", type=float, default=None, help="Negative prismatic peak magnitude in meters")
parser.add_argument("--prismatic-frequency", type=float, default=5, help="Prismatic sine frequency in Hz")
parser.add_argument("--prismatic-slew-rate", type=float, default=0.5, help="Max m/s command slew")
parser.add_argument("--k-amplitude", type=float, default=0.01, help="K-link extension amplitude in meters for all legs")
parser.add_argument("--f-amplitude", type=float, default=None, help="F-link extension amplitude in meters for all legs")
parser.add_argument("--h-amplitude", type=float, default=None, help="H-link extension amplitude in meters for all legs")
parser.add_argument("--k-positive-amplitude", type=float, default=0.01, help="K-link positive peak in meters for all legs")
parser.add_argument("--f-positive-amplitude", type=float, default=0.002, help="F-link positive peak in meters for all legs")
parser.add_argument("--h-positive-amplitude", type=float, default=0.01, help="H-link positive peak in meters for all legs")
parser.add_argument("--k-negative-amplitude", type=float, default=0, help="K-link negative peak magnitude in meters for all legs")
parser.add_argument("--f-negative-amplitude", type=float, default=0, help="F-link negative peak magnitude in meters for all legs")
parser.add_argument("--h-negative-amplitude", type=float, default=0.01, help="H-link negative peak magnitude in meters for all legs")
parser.add_argument("--k-frequency", type=float, default=None, help="K-link sine frequency in Hz for all legs")
parser.add_argument("--f-frequency", type=float, default=None, help="F-link sine frequency in Hz for all legs")
parser.add_argument("--h-frequency", type=float, default=None, help="H-link sine frequency in Hz for all legs")
parser.add_argument("--k-slew-rate", type=float, default=None, help="K-link max m/s command slew for all legs")
parser.add_argument("--f-slew-rate", type=float, default=None, help="F-link max m/s command slew for all legs")
parser.add_argument("--h-slew-rate", type=float, default=None, help="H-link max m/s command slew for all legs")
parser.add_argument("--reach-tolerance", type=float, default=1e-4, help="Flip command when all group joints are within this error (m)")
parser.add_argument("--crank-run-speed", type=float, default=4, help="Post-settle crank speed command magnitude (rad/s)")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path("jansen_assembly_rolling_hills.xml")
data = mujoco.MjData(model)

# Crank actuators and crank joints for settle phase.
crank_act_names = ["crank_motor", "crank_motor_r", "crank_motor_y", "crank_motor_r_y"]
crank_joint_names = ["hinge_m", "hinge_m_r", "hinge_m_y", "hinge_m_r_y"]
crank_act_ids = [get_actuator_id(model, n, required=True) for n in crank_act_names]
crank_joint_ids = [get_joint_id(model, n, required=True) for n in crank_joint_names]
crank_qadrs = [model.jnt_qposadr[jid] for jid in crank_joint_ids]
crank_dofadrs = [model.jnt_dofadr[jid] for jid in crank_joint_ids]

# Per-leg prismatic actuators. h2..h4 may not exist in current XML.
k_act_ids = [get_actuator_id(model, n, required=True) for n in ["k1_prismatic", "k2_prismatic", "k3_prismatic", "k4_prismatic"]]
f_act_ids = [get_actuator_id(model, n, required=True) for n in ["f1_prismatic", "f2_prismatic", "f3_prismatic", "f4_prismatic"]]
h_names = ["h1_prismatic", "h2_prismatic", "h3_prismatic", "h4_prismatic"]
h_act_ids = [get_actuator_id(model, n, required=False) for n in h_names]
k_joint_names = ["slide_k", "slide_k_r", "slide_k_y", "slide_k_r_y"]
f_joint_names = ["slide_f", "slide_f_r", "slide_f_y", "slide_f_r_y"]
h_joint_names = ["slide_h", "slide_h_r", "slide_h_y", "slide_h_r_y"]
k_joint_ids = [get_joint_id(model, n, required=True) for n in k_joint_names]
f_joint_ids = [get_joint_id(model, n, required=True) for n in f_joint_names]
h_joint_ids = [get_joint_id(model, n, required=False) for n in h_joint_names]
k_qadrs = [model.jnt_qposadr[jid] for jid in k_joint_ids]
f_qadrs = [model.jnt_qposadr[jid] for jid in f_joint_ids]
h_qadrs = [model.jnt_qposadr[jid] if jid >= 0 else -1 for jid in h_joint_ids]

plate_free_jid = get_joint_id(model, "plate_free", required=True)
plate_qadr = model.jnt_qposadr[plate_free_jid]
plate_vadr = model.jnt_dofadr[plate_free_jid]
plate_qpos0 = data.qpos[plate_qadr:plate_qadr + 7].copy()
if args.lock_air:
    plate_qpos0[2] = float(args.lock_air_z)

# Damping schedule for hinge joints.
settle_damping = 5.0
run_damping = 0.5
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
    for aid in crank_act_ids:
        data.ctrl[aid] = 0.0


def apply_camera(cam) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if args.cam_azimuth is not None:
        cam.azimuth = float(args.cam_azimuth)
    if args.cam_elevation is not None:
        cam.elevation = float(args.cam_elevation)
    if args.cam_distance is not None:
        cam.distance = float(args.cam_distance)
    cam.lookat[:] = [args.cam_lookat_x, args.cam_lookat_y, args.cam_lookat_z]


def set_prismatic_ctrl(act_ids, states) -> None:
    for i, aid in enumerate(act_ids):
        if aid >= 0:
            data.ctrl[aid] = states[i]


set_hinge_damping(settle_damping)
damping_switched = False
position_targets_reached = False
if args.lock_air:
    position_targets_reached = True

idle_settle_time = args.idle_settle_time
position_settle_time = args.position_settle_time
min_settle_end_time = idle_settle_time + position_settle_time

position_kp = 20.0
position_max_cmd = 25.0
position_tol = 0.01
velocity_tol = 0.05

target_q = [float(data.qpos[qadr]) for qadr in crank_qadrs]
target_q[0] -= math.pi / 2  # red: -90 deg
target_q[1] += math.pi / 2  # green: +90 deg
target_q[2] += math.pi / 2  # blue: +90 deg
target_q[3] -= math.pi / 2  # yellow: -90 deg

k_states = [0.0, 0.0, 0.0, 0.0]
f_states = [0.0, 0.0, 0.0, 0.0]
h_states = [0.0, 0.0, 0.0, 0.0]
run_phase_started = False
base_amp = abs(float(args.prismatic_amplitude))
base_positive_amp = (
    abs(float(args.prismatic_positive_amplitude))
    if args.prismatic_positive_amplitude is not None
    else base_amp
)
base_negative_amp = (
    abs(float(args.prismatic_negative_amplitude))
    if args.prismatic_negative_amplitude is not None
    else base_amp
)
base_frequency = abs(float(args.prismatic_frequency))
base_slew_rate = abs(float(args.prismatic_slew_rate))


def resolve_amplitude_limits(symmetric_arg, positive_arg, negative_arg):
    symmetric_amp = abs(float(symmetric_arg)) if symmetric_arg is not None else None
    positive_default = symmetric_amp if symmetric_amp is not None else base_positive_amp
    negative_default = symmetric_amp if symmetric_amp is not None else base_negative_amp
    positive_amp = abs(float(positive_arg)) if positive_arg is not None else positive_default
    negative_amp = abs(float(negative_arg)) if negative_arg is not None else negative_default
    return positive_amp, negative_amp


group_positive_amp = {}
group_negative_amp = {}
group_positive_amp["k"], group_negative_amp["k"] = resolve_amplitude_limits(
    args.k_amplitude,
    args.k_positive_amplitude,
    args.k_negative_amplitude,
)
group_positive_amp["f"], group_negative_amp["f"] = resolve_amplitude_limits(
    args.f_amplitude,
    args.f_positive_amplitude,
    args.f_negative_amplitude,
)
group_positive_amp["h"], group_negative_amp["h"] = resolve_amplitude_limits(
    args.h_amplitude,
    args.h_positive_amplitude,
    args.h_negative_amplitude,
)
group_frequency = {
    "k": abs(float(args.k_frequency)) if args.k_frequency is not None else base_frequency,
    "f": abs(float(args.f_frequency)) if args.f_frequency is not None else base_frequency,
    "h": abs(float(args.h_frequency)) if args.h_frequency is not None else base_frequency,
}
group_slew_rate = {
    "k": abs(float(args.k_slew_rate)) if args.k_slew_rate is not None else base_slew_rate,
    "f": abs(float(args.f_slew_rate)) if args.f_slew_rate is not None else base_slew_rate,
    "h": abs(float(args.h_slew_rate)) if args.h_slew_rate is not None else base_slew_rate,
}
crank_run_speed = abs(float(args.crank_run_speed))
run_phase_start_time = 0.0


def all_prismatic_reached(target_cmd: float, tol: float) -> bool:
    for i in range(4):
        if abs(float(data.qpos[k_qadrs[i]]) - target_cmd) > tol:
            return False
        if abs(float(data.qpos[f_qadrs[i]]) - target_cmd) > tol:
            return False
        if h_qadrs[i] >= 0 and abs(float(data.qpos[h_qadrs[i]]) - target_cmd) > tol:
            return False
    return True


def slew_states(states, target_cmd: float, slew_rate: float, dt: float) -> None:
    max_delta = abs(slew_rate) * dt
    for i in range(len(states)):
        if max_delta <= 0.0:
            states[i] = target_cmd
        else:
            delta = clamp(target_cmd - states[i], -max_delta, max_delta)
            states[i] += delta


def asymmetric_sine(phase: float, positive_amp: float, negative_amp: float) -> float:
    wave = math.sin(phase)
    if wave >= 0.0:
        return positive_amp * wave
    return negative_amp * wave


def simulation_step() -> None:
    global position_targets_reached
    global damping_switched
    global run_phase_started
    global run_phase_start_time

    sim_time = data.time
    idle_phase = sim_time < idle_settle_time
    position_settle_phase = (not idle_phase) and (not position_targets_reached)
    run_phase = not (idle_phase or position_settle_phase)

    if idle_phase:
        lock_plate_to_initial_pose()
        zero_all_cranks()
    elif position_settle_phase:
        lock_plate_to_initial_pose()
        zero_all_cranks()

        all_reached = True
        for i, aid in enumerate(crank_act_ids):
            lock_plate_to_initial_pose()
            err = target_q[i] - float(data.qpos[crank_qadrs[i]])
            data.ctrl[aid] = clamp(position_kp * err, -position_max_cmd, position_max_cmd)
            reached = abs(err) <= position_tol and abs(float(data.qvel[crank_dofadrs[i]])) <= velocity_tol
            all_reached = all_reached and reached

        if sim_time >= min_settle_end_time and all_reached:
            position_targets_reached = True
    else:
        # lock_plate_to_initial_pose()
        if not damping_switched:
            set_hinge_damping(run_damping)
            damping_switched = True
        if not run_phase_started:
            run_phase_started = True
            run_phase_start_time = sim_time

        # After settle: red+green CW, blue+yellow CCW.
        data.ctrl[crank_act_ids[0]] = crank_run_speed  # red
        data.ctrl[crank_act_ids[1]] = crank_run_speed  # green
        data.ctrl[crank_act_ids[2]] = crank_run_speed   # blue
        data.ctrl[crank_act_ids[3]] = crank_run_speed   # yellow

        run_time = max(0.0, sim_time - run_phase_start_time)
        k_cmd = asymmetric_sine(
            2.0 * math.pi * group_frequency["k"] * run_time,
            group_positive_amp["k"],
            group_negative_amp["k"],
        )
        f_cmd = asymmetric_sine(
            2.0 * math.pi * group_frequency["f"] * run_time,
            group_positive_amp["f"],
            group_negative_amp["f"],
        )
        h_cmd = asymmetric_sine(
            2.0 * math.pi * group_frequency["h"] * run_time,
            group_positive_amp["h"],
            group_negative_amp["h"],
        )

        slew_states(k_states, k_cmd, group_slew_rate["k"], model.opt.timestep)
        slew_states(f_states, f_cmd, group_slew_rate["f"], model.opt.timestep)
        slew_states(h_states, h_cmd, group_slew_rate["h"], model.opt.timestep)

        set_prismatic_ctrl(k_act_ids, k_states)
        set_prismatic_ctrl(f_act_ids, f_states)
        set_prismatic_ctrl(h_act_ids, h_states)

    mujoco.mj_step(model, data)
    if args.lock_air or idle_phase or position_settle_phase:
        lock_plate_to_initial_pose()
    mujoco.mj_forward(model, data)


if args.video:
    video_dir = os.path.dirname(args.video) or "."
    os.makedirs(video_dir, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    apply_camera(cam)

    frame_skip = args.video_frame_skip
    if frame_skip <= 0:
        frame_skip = max(1, int(round(1.0 / (args.video_fps * model.opt.timestep))))

    frames_written = 0
    with imageio.get_writer(args.video, fps=args.video_fps, codec="libx264") as writer:
        for step_idx in range(args.video_steps):
            simulation_step()
            if step_idx % frame_skip == 0:
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
                frames_written += 1
    renderer.close()
    print(
        f"Video saved: {args.video} | resolution={args.video_width}x{args.video_height} | "
        f"fps={args.video_fps} | frame_skip={frame_skip} | frames={frames_written}"
    )
else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_camera(viewer.cam)
        wall_start = time.perf_counter()
        while viewer.is_running():
            simulation_step()
            viewer.sync()
            target_wall_time = wall_start + data.time
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
