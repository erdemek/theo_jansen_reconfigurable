import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class JansenEnv(gym.Env):
    """
    Jansen Walker Environment - Active RL Control.
    The agent independently controls 4 k-link, 4 f-link and 4 h-link targets.
    Internal slew rate ensures physics stability at 800,000 kp.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(
        self,
        xml_path="jansen_assembly_red_articulated.xml",
        render_mode=None,
        prismatic_cmd_limit=0.005,
        k_positive_amplitude=None,
        f_positive_amplitude=None,
        h_positive_amplitude=None,
        k_negative_amplitude=None,
        f_negative_amplitude=None,
        h_negative_amplitude=None,
        k_slew_rate=0.05,
        reset_settle_time=6.2,
    ):
        super(JansenEnv, self).__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Physics timing (0.00025s) -> RL timing (0.0025s)
        self.frame_skip = 10 
        self.dt = self.model.opt.timestep * self.frame_skip

        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # --- IDs ---
        def get_required_actuator(name):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Missing actuator '{name}' in XML: {xml_path}")
            return aid

        def get_required_joint(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Missing joint '{name}' in XML: {xml_path}")
            return jid

        self.motor_ids = [get_required_actuator(n) for n in ["crank_motor", "crank_motor_r", "crank_motor_y", "crank_motor_r_y"]]
        self.k_act_ids = [get_required_actuator(n) for n in ["k1_prismatic", "k2_prismatic", "k3_prismatic", "k4_prismatic"]]
        self.f_act_ids = [get_required_actuator(n) for n in ["f1_prismatic", "f2_prismatic", "f3_prismatic", "f4_prismatic"]]
        self.h_act_ids = [get_required_actuator(n) for n in ["h1_prismatic", "h2_prismatic", "h3_prismatic", "h4_prismatic"]]

        self.crank_joint_qadrs = [self.model.jnt_qposadr[get_required_joint(n)] for n in ["hinge_m", "hinge_m_r", "hinge_m_y", "hinge_m_r_y"]]
        self.slide_k_qadrs = [self.model.jnt_qposadr[get_required_joint(n)] for n in ["slide_k", "slide_k_r", "slide_k_y", "slide_k_r_y"]]
        self.slide_f_qadrs = [self.model.jnt_qposadr[get_required_joint(n)] for n in ["slide_f", "slide_f_r", "slide_f_y", "slide_f_r_y"]]
        self.slide_h_qadrs = [self.model.jnt_qposadr[get_required_joint(n)] for n in ["slide_h", "slide_h_r", "slide_h_y", "slide_h_r_y"]]
        
        plate_free_jid = get_required_joint("plate_free")
        self.plate_qadr = self.model.jnt_qposadr[plate_free_jid]
        self.plate_vadr = self.model.jnt_dofadr[plate_free_jid]

        self.hinge_dof_ids = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.hinge_dof_ids.append(self.model.jnt_dofadr[j])

        # Action: target k/f/h extensions in normalized [-1, 1] for 4 legs (12 dim total).
        # Each action is linearly mapped across that link group's min/max limits.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Obs: Vel(6), Up(3), CrankPhase(8), k(4), f(4), h(4) = 29 dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        self.k_slew_rate = float(k_slew_rate)
        self.prismatic_cmd_limit = float(prismatic_cmd_limit)
        k_positive_amplitude = self.prismatic_cmd_limit if k_positive_amplitude is None else abs(float(k_positive_amplitude))
        f_positive_amplitude = self.prismatic_cmd_limit if f_positive_amplitude is None else abs(float(f_positive_amplitude))
        h_positive_amplitude = self.prismatic_cmd_limit if h_positive_amplitude is None else abs(float(h_positive_amplitude))
        k_negative_amplitude = self.prismatic_cmd_limit if k_negative_amplitude is None else abs(float(k_negative_amplitude))
        f_negative_amplitude = self.prismatic_cmd_limit if f_negative_amplitude is None else abs(float(f_negative_amplitude))
        h_negative_amplitude = self.prismatic_cmd_limit if h_negative_amplitude is None else abs(float(h_negative_amplitude))
        self.prismatic_min_limits = -np.array(
            [k_negative_amplitude] * 4 + [f_negative_amplitude] * 4 + [h_negative_amplitude] * 4,
            dtype=np.float32,
        )
        self.prismatic_max_limits = np.array(
            [k_positive_amplitude] * 4 + [f_positive_amplitude] * 4 + [h_positive_amplitude] * 4,
            dtype=np.float32,
        )
        self.prismatic_mid_targets = 0.5 * (self.prismatic_min_limits + self.prismatic_max_limits)
        self.roll_penalty_gain = 30.0
        self.roll_rate_penalty_gain = 4.0
        self.lateral_velocity_penalty_gain = 5.0
        self.heading_y_penalty_gain = 2.0
        self.prismatic_target_change_penalty_gain = 10.0
        self.speed_cmd = 3.0
        self.max_steps = 10000
        self.reset_settle_time = float(reset_settle_time)
        self.current_step = 0
        self.k_extension_states = self.prismatic_mid_targets[0:4].copy()
        self.f_extension_states = self.prismatic_mid_targets[4:8].copy()
        self.h_extension_states = self.prismatic_mid_targets[8:12].copy()
        self.prev_action = self.prismatic_mid_targets.copy()
        self.real_time_start = 0

    def _action_to_prismatic_targets(self, action):
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action_01 = 0.5 * (clipped_action + 1.0)
        return self.prismatic_min_limits + action_01 * (self.prismatic_max_limits - self.prismatic_min_limits)

    def _apply_prismatic_ctrl(self):
        for i in range(4):
            self.data.ctrl[self.k_act_ids[i]] = self.k_extension_states[i]
            self.data.ctrl[self.f_act_ids[i]] = self.f_extension_states[i]
            self.data.ctrl[self.h_act_ids[i]] = self.h_extension_states[i]

    def set_hinge_damping(self, value):
        for dof_id in self.hinge_dof_ids:
            self.model.dof_damping[dof_id] = value

    def _get_obs(self):
        # 1. Velocities
        qvel = self.data.qvel[self.plate_vadr:self.plate_vadr+6]
        # 2. Orientation (Up vector)
        plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate_body")
        up_vector = self.data.xmat[plate_body_id][6:9]
        # 3. Crank Phases (sin/cos)
        crank_obs = []
        for adr in self.crank_joint_qadrs:
            angle = float(self.data.qpos[adr])
            crank_obs.extend([math.sin(angle), math.cos(angle)])
        # 4. Current k/f/h states
        k_exts = self.k_extension_states * 100.0 # Scale for NN
        f_exts = self.f_extension_states * 100.0 # Scale for NN
        h_exts = self.h_extension_states * 100.0 # Scale for NN
        return np.concatenate([qvel, up_vector, crank_obs, k_exts, f_exts, h_exts]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.set_hinge_damping(5.0) 
        
        target_q_m = float(self.data.qpos[self.crank_joint_qadrs[0]] + math.pi)
        target_q_r_y = float(self.data.qpos[self.crank_joint_qadrs[3]] + math.pi)
        plate_qpos0 = self.data.qpos[self.plate_qadr : self.plate_qadr + 7].copy()
        
        # Reset settle period (shorter values are much faster but can be less stable).
        for _ in range(int(self.reset_settle_time / self.model.opt.timestep)):
            self.data.qpos[self.plate_qadr : self.plate_qadr + 7] = plate_qpos0
            self.data.qvel[self.plate_vadr : self.plate_vadr + 6] = 0.0
            self.data.ctrl[self.motor_ids[0]] = 20.0 * (target_q_m - self.data.qpos[self.crank_joint_qadrs[0]])
            self.data.ctrl[self.motor_ids[3]] = 20.0 * (target_q_r_y - self.data.qpos[self.crank_joint_qadrs[3]])
            mujoco.mj_step(self.model, self.data)
            
        self.damping_switched = False
        self.k_extension_states = self.prismatic_mid_targets[0:4].copy()
        self.f_extension_states = self.prismatic_mid_targets[4:8].copy()
        self.h_extension_states = self.prismatic_mid_targets[8:12].copy()
        self.prev_action = self.prismatic_mid_targets.copy()
        self._apply_prismatic_ctrl()
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        self.real_time_start = time.perf_counter()
        return self._get_obs(), {}

    def step(self, action):
        if not self.damping_switched:
            self.set_hinge_damping(0.5) 
            self.damping_switched = True
        
        # Split 12 normalized actions into k/f/h targets with asymmetric min/max limits.
        prismatic_targets = self._action_to_prismatic_targets(action)
        target_k = prismatic_targets[0:4]
        target_f = prismatic_targets[4:8]
        target_h = prismatic_targets[8:12]

        x_before = self.data.qpos[self.plate_qadr]
        
        for _ in range(self.frame_skip):
            for mid in self.motor_ids: self.data.ctrl[mid] = self.speed_cmd
            
            # Independent slew control for 4 k-links, 4 f-links and 4 h-links
            max_delta = self.k_slew_rate * self.model.opt.timestep
            self.k_extension_states += np.clip(target_k - self.k_extension_states, -max_delta, max_delta)
            self.f_extension_states += np.clip(target_f - self.f_extension_states, -max_delta, max_delta)
            self.h_extension_states += np.clip(target_h - self.h_extension_states, -max_delta, max_delta)
            
            self._apply_prismatic_ctrl()
            
            mujoco.mj_step(self.model, self.data)
            
            if self.render_mode == "human" and self.viewer:
                self.viewer.sync()
                # Real-time sleep if requested
                target_wall = self.real_time_start + self.data.time
                sleep_for = target_wall - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
        
        obs = self._get_obs()
        up_y = obs[7]
        up_z = obs[8]
        vx, vy, vz = obs[0:3]
        wx, wy, wz = obs[3:6]
        plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate_body")
        heading_y = float(self.data.xmat[plate_body_id][1])
        
        # Forward reward
        reward = (x_before - self.data.qpos[self.plate_qadr]) * 1000.0 
        
        # FLUCTUATION PUNISHMENT (Aggressive)
        reward -= abs(vz) * 10.0  # Bobbing (was 2.0)
        reward -= abs(vy) * self.lateral_velocity_penalty_gain
        
        # TILT PUNISHMENT (Keep body level)
        # up_z is obs[8], perfectly vertical is 1.0
        reward -= (1.0 - up_z) * 20.0 
        # Explicit roll suppression: up_y departs from 0 as body rolls.
        reward -= abs(up_y) * self.roll_penalty_gain
        # Roll/pitch rate damping to reduce rocking dynamics.
        reward -= (abs(wx) + abs(wy)) * self.roll_rate_penalty_gain
        reward -= abs(heading_y) * self.heading_y_penalty_gain
        reward -= self.prismatic_target_change_penalty_gain * float(np.sum(np.abs(prismatic_targets - self.prev_action)))
        self.prev_action = prismatic_targets.copy()
        
        reward += 0.5            # Survival
        
        terminated = up_z < 0.6 or self.data.qpos[self.plate_qadr+2] < 0.05
        if terminated: reward -= 500.0 # Heavier fall penalty
        
        self.current_step += 1
        return obs, float(reward), terminated, self.current_step >= self.max_steps, {}

    def close(self):
        if self.viewer: self.viewer.close()
