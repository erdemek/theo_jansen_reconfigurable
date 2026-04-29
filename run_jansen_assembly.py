import argparse
import math
import os
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer


parser = argparse.ArgumentParser(description="Run Jansen assembly simulation")
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
parser.add_argument("--focus-rear-left-leg", action="store_true", help="Use a close-up camera preset for rear-left leg")
args = parser.parse_args()

if args.focus_rear_left_leg:
    if args.cam_azimuth is None:
        args.cam_azimuth = 145.0
    if args.cam_elevation is None:
        args.cam_elevation = -18.0
    if args.cam_distance is None:
        args.cam_distance = 0.34
    if args.cam_lookat_x == 0.0 and args.cam_lookat_y == 0.0 and args.cam_lookat_z == 0.12:
        args.cam_lookat_x = -0.07
        args.cam_lookat_y = 0.05
        args.cam_lookat_z = 0.12

model = mujoco.MjModel.from_xml_path("jansen_assembly_red_articulated_rocky_hfield.xml")
data = mujoco.MjData(model)

# Drive all crank joints using velocity actuators.
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")
act_id_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_r")
act_id_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_y")
act_id_r_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor_r_y")
if min(act_id, act_id_r, act_id_y, act_id_r_y) < 0:
    raise RuntimeError("One or more crank actuators were not found in jansen_assembly.xml")

# Prismatic actuators for k-link reconfiguration.
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

# Crank joints used for settle-time position targeting.
jid_m = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_m")
jid_m_r_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_m_r_y")
if min(jid_m, jid_m_r_y) < 0:
    raise RuntimeError("One or more crank joints were not found in jansen_assembly.xml")
qadr_m = model.jnt_qposadr[jid_m]
qadr_m_r_y = model.jnt_qposadr[jid_m_r_y]
dof_m = model.jnt_dofadr[jid_m]
dof_m_r_y = model.jnt_dofadr[jid_m_r_y]

# Plate is free, but we pin it during settling and release afterward.
plate_free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
if plate_free_jid < 0:
    raise RuntimeError("Joint 'plate_free' not found. Check jansen_assembly.xml")
plate_qadr = model.jnt_qposadr[plate_free_jid]
plate_vadr = model.jnt_dofadr[plate_free_jid]
plate_qpos0 = data.qpos[plate_qadr:plate_qadr + 7].copy()
if args.lock_air:
    plate_qpos0[2] = float(args.lock_air_z)

# Damping schedule for all hinge DOFs.
settle_damping = 5
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
    data.ctrl[act_id] = 0.0
    data.ctrl[act_id_r] = 0.0
    data.ctrl[act_id_y] = 0.0
    data.ctrl[act_id_r_y] = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def set_k1_extension(ext_cmd: float) -> None:
    val = clamp(ext_cmd, -0.005, 0.005)
    data.ctrl[k1_prismatic_act_id] = val
    data.ctrl[k2_prismatic_act_id] = val
    data.ctrl[k3_prismatic_act_id] = val
    data.ctrl[k4_prismatic_act_id] = val


def apply_camera(cam) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if args.cam_azimuth is not None:
        cam.azimuth = float(args.cam_azimuth)
    if args.cam_elevation is not None:
        cam.elevation = float(args.cam_elevation)
    if args.cam_distance is not None:
        cam.distance = float(args.cam_distance)
    cam.lookat[:] = [args.cam_lookat_x, args.cam_lookat_y, args.cam_lookat_z]


# Run-phase velocity command (rad/s).
speed_cmd = 5
initial_body_vx = -0.3 * 0
real_time_scale = 1.0
k1_extension_cmd = -0.0035
k1_extension_state = 0.0
k1_slew_rate = 0.005

# Settle schedule: 0.2 s idle then position-controlled pre-rotation.
idle_settle_time = args.idle_settle_time
position_settle_time = args.position_settle_time
min_settle_end_time = idle_settle_time + position_settle_time

# Position-servo parameters for pre-rotation to 180 deg.
target_delta = math.pi
position_kp = 20.0
position_max_cmd = 25.0
position_tol = 0.01
velocity_tol = 0.05

# Diagonal pair: rear-right + front-left in this model mapping.
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

# Initialize slew state to whatever the joint currently is
k1_extension_state = float(data.qpos[slide_k_qadr])
run_phase_started = False
crank_angle_at_run_start = 0.0
k1_triggered = False


def simulation_step() -> None:
    global position_targets_reached
    global damping_switched
    global initial_velocity_applied
    global run_phase_started
    global crank_angle_at_run_start
    global k1_triggered
    global k1_extension_cmd
    global k1_extension_state

    sim_time = data.time
    idle_phase = sim_time < idle_settle_time
    position_settle_phase = (not idle_phase) and (not position_targets_reached)

    if idle_phase:
        lock_plate_to_initial_pose()
        zero_all_cranks()
        max_delta = k1_slew_rate * model.opt.timestep
        k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
        set_k1_extension(k1_extension_state)
    elif position_settle_phase:
        lock_plate_to_initial_pose()
        zero_all_cranks()
        max_delta = k1_slew_rate * model.opt.timestep
        k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
        set_k1_extension(k1_extension_state)

        err_m = target_q_m - float(data.qpos[qadr_m])
        err_m_r_y = target_q_m_r_y - float(data.qpos[qadr_m_r_y])
        data.ctrl[stagger_act_1] = clamp(position_kp * err_m, -position_max_cmd, position_max_cmd)
        data.ctrl[stagger_act_2] = clamp(position_kp * err_m_r_y, -position_max_cmd, position_max_cmd)

        reached_m = abs(err_m) <= position_tol and abs(float(data.qvel[dof_m])) <= velocity_tol
        reached_m_r_y = abs(err_m_r_y) <= position_tol and abs(float(data.qvel[dof_m_r_y])) <= velocity_tol
        if sim_time >= min_settle_end_time and reached_m and reached_m_r_y:
            position_targets_reached = True
    else:
        # lock_plate_to_initial_pose()
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
            k1_extension_cmd = 0.0035 * 0
            k1_triggered = True

        # if k1_triggered:
        #     k1_tol = 0.0001
        #     k1_actual = float(data.qpos[slide_k_qadr])
        #     if abs(k1_actual - k1_extension_cmd) <= k1_tol:
        #         k1_extension_cmd = -k1_extension_cmd

        data.ctrl[act_id] = speed_cmd
        data.ctrl[act_id_r] = speed_cmd
        data.ctrl[act_id_y] = speed_cmd
        data.ctrl[act_id_r_y] = speed_cmd

        max_delta = k1_slew_rate * model.opt.timestep
        k1_extension_state += clamp(k1_extension_cmd - k1_extension_state, -max_delta, max_delta)
        set_k1_extension(k1_extension_state)

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
        # Choose a skip that keeps playback close to real-time simulation speed.
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
            target_wall_time = wall_start + (data.time / real_time_scale)
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
