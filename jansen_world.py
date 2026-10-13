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

    # Runtime-only model tuning. The baseline profile leaves the compiled XML
    # unchanged. Explicit constructor/CLI values override profile values.
    STABILITY_PROFILES = {
        "baseline": {},
        "dt2ms_soft": {
            "prismatic_kp_scale": 0.10,
            "equality_time_constant": 0.020,
            "prismatic_limit_time_constant": 0.020,
            "minimum_body_mass": 0.001,
            "contact_time_constant": 0.020,
            "crank_kv": 10.0,
            "prismatic_force_limit": 1000.0,
            "crank_speed": 3.0,
            "reset_crank_gain": 10.0,
        },
    }

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
        prismatic_max_acceleration=0.0,
        reset_settle_time=6.2,
        train_prismatic_groups="kfh",
        reward_preset="h_only_straight_walk",
        physics_timestep=None,
        control_timestep=None,
        episode_seconds=None,
        stability_profile="baseline",
        prismatic_kp_scale=None,
        equality_time_constant=None,
        prismatic_limit_time_constant=None,
        minimum_body_mass=None,
        contact_time_constant=None,
        crank_kv=None,
        prismatic_force_limit=None,
        crank_speed=None,
        reset_crank_gain=None,
    ):
        super(JansenEnv, self).__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        if physics_timestep is not None:
            physics_timestep = float(physics_timestep)
            if physics_timestep <= 0.0:
                raise ValueError("physics_timestep must be > 0")
            self.model.opt.timestep = physics_timestep

        physics_timestep = float(self.model.opt.timestep)
        if control_timestep is None:
            # Preserve the original 0.25 ms x 10 = 2.5 ms behavior.
            self.frame_skip = 10
        else:
            control_timestep = float(control_timestep)
            if control_timestep <= 0.0:
                raise ValueError("control_timestep must be > 0")
            self.frame_skip = int(round(control_timestep / physics_timestep))
            if self.frame_skip < 1 or not math.isclose(
                control_timestep,
                self.frame_skip * physics_timestep,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "control_timestep must be an integer multiple of the "
                    f"physics timestep ({physics_timestep:g} s)"
                )

        self.dt = self.model.opt.timestep * self.frame_skip

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

        profile_name = str(stability_profile).lower()
        if profile_name not in self.STABILITY_PROFILES:
            raise ValueError(
                "stability_profile must be one of: "
                f"{sorted(self.STABILITY_PROFILES)}"
            )
        profile = self.STABILITY_PROFILES[profile_name]

        def tuned_value(name, explicit_value, baseline_value=None):
            if explicit_value is not None:
                return explicit_value
            return profile.get(name, baseline_value)

        kp_scale = float(tuned_value("prismatic_kp_scale", prismatic_kp_scale, 1.0))
        equality_time_constant = tuned_value(
            "equality_time_constant", equality_time_constant
        )
        prismatic_limit_time_constant = tuned_value(
            "prismatic_limit_time_constant", prismatic_limit_time_constant
        )
        minimum_body_mass = tuned_value("minimum_body_mass", minimum_body_mass)
        contact_time_constant = tuned_value(
            "contact_time_constant", contact_time_constant
        )
        crank_kv = tuned_value("crank_kv", crank_kv)
        prismatic_force_limit = tuned_value(
            "prismatic_force_limit", prismatic_force_limit
        )
        self.speed_cmd = float(tuned_value("crank_speed", crank_speed, 3.0))
        self.reset_crank_gain = float(
            tuned_value("reset_crank_gain", reset_crank_gain, 20.0)
        )

        positive_values = {
            "prismatic_kp_scale": kp_scale,
            "equality_time_constant": equality_time_constant,
            "prismatic_limit_time_constant": prismatic_limit_time_constant,
            "minimum_body_mass": minimum_body_mass,
            "contact_time_constant": contact_time_constant,
            "crank_kv": crank_kv,
            "prismatic_force_limit": prismatic_force_limit,
            "reset_crank_gain": self.reset_crank_gain,
        }
        for name, value in positive_values.items():
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.speed_cmd < 0.0:
            raise ValueError("crank_speed must be >= 0")

        prismatic_actuator_ids = self.k_act_ids + self.f_act_ids + self.h_act_ids
        for actuator_id in prismatic_actuator_ids:
            kp = float(self.model.actuator_gainprm[actuator_id, 0]) * kp_scale
            self.model.actuator_gainprm[actuator_id, 0] = kp
            self.model.actuator_biasprm[actuator_id, 1] = -kp
            if prismatic_force_limit is not None:
                force_limit = float(prismatic_force_limit)
                self.model.actuator_forcelimited[actuator_id] = 1
                self.model.actuator_forcerange[actuator_id] = [
                    -force_limit,
                    force_limit,
                ]

        if crank_kv is not None:
            crank_kv = float(crank_kv)
            for actuator_id in self.motor_ids:
                self.model.actuator_gainprm[actuator_id, 0] = crank_kv
                self.model.actuator_biasprm[actuator_id, 2] = -crank_kv

        if equality_time_constant is not None and self.model.neq:
            self.model.eq_solref[:, 0] = float(equality_time_constant)

        if prismatic_limit_time_constant is not None:
            for joint_id in range(self.model.njnt):
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                    self.model.jnt_solref[joint_id, 0] = float(
                        prismatic_limit_time_constant
                    )

        if contact_time_constant is not None:
            self.model.geom_solref[:, 0] = float(contact_time_constant)

        adjusted_body_count = 0
        if minimum_body_mass is not None:
            minimum_body_mass = float(minimum_body_mass)
            for body_id in range(1, self.model.nbody):
                old_mass = float(self.model.body_mass[body_id])
                if 0.0 < old_mass < minimum_body_mass:
                    mass_ratio = minimum_body_mass / old_mass
                    self.model.body_mass[body_id] = minimum_body_mass
                    self.model.body_inertia[body_id] *= mass_ratio
                    adjusted_body_count += 1

        self.data = mujoco.MjData(self.model)
        if adjusted_body_count:
            mujoco.mj_setConst(self.model, self.data)

        self.stability_profile = profile_name
        self.physics_tuning = {
            "prismatic_kp_scale": kp_scale,
            "equality_time_constant": equality_time_constant,
            "prismatic_limit_time_constant": prismatic_limit_time_constant,
            "minimum_body_mass": minimum_body_mass,
            "adjusted_body_count": adjusted_body_count,
            "contact_time_constant": contact_time_constant,
            "crank_kv": crank_kv,
            "prismatic_force_limit": prismatic_force_limit,
            "crank_speed": self.speed_cmd,
            "reset_crank_gain": self.reset_crank_gain,
        }

        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.train_prismatic_groups = "".join(
            group for group in "kfh" if group in str(train_prismatic_groups).lower()
        )
        if not self.train_prismatic_groups:
            raise ValueError("train_prismatic_groups must include at least one of: k, f, h")
        self.reward_preset = str(reward_preset).lower()
        valid_reward_presets = {
            "h_only_straight_walk",
            "h_only_straight_walk_survival_2_action_penalty",
            "h_only_straight_walk_normalized_fullscale",
        }
        if self.reward_preset not in valid_reward_presets:
            raise ValueError(f"reward_preset must be one of: {sorted(valid_reward_presets)}")
        self.train_prismatic_indices = []
        for group in self.train_prismatic_groups:
            if group == "k":
                self.train_prismatic_indices.extend(range(0, 4))
            elif group == "f":
                self.train_prismatic_indices.extend(range(4, 8))
            elif group == "h":
                self.train_prismatic_indices.extend(range(8, 12))
        self.train_prismatic_indices = np.array(self.train_prismatic_indices, dtype=np.int32)

        # Action: normalized [-1, 1] targets for selected k/f/h groups.
        # Untrained prismatic groups are held at zero target.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.train_prismatic_indices),), dtype=np.float32)
        
        # Obs: Vel(6), Up(3), CrankPhase(8), k(4), f(4), h(4) = 29 dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        self.k_slew_rate = float(k_slew_rate)
        # Post-PPO29 trial: optional acceleration limiting was introduced after
        # PPO29 was trained. Keep 0.0 as the default to reproduce PPO29 exactly.
        self.prismatic_max_acceleration = float(prismatic_max_acceleration)
        if self.prismatic_max_acceleration < 0.0:
            raise ValueError("prismatic_max_acceleration must be >= 0")
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
        self.initial_prismatic_targets = np.zeros(12, dtype=np.float32)
        self.initial_prismatic_targets[self.train_prismatic_indices] = self.prismatic_mid_targets[self.train_prismatic_indices]
        if self.reward_preset == "h_only_straight_walk_survival_2_action_penalty":
            self.fall_penalty = 0.0
            self.survival_reward = 0.0
            self.forward_progress_reward_gain = 2000.0
            self.forward_speed_reward_gain = 0.0
            self.roll_pitch_rate_penalty_gain = 0.0
            self.side_roll_penalty_gain = 0.0
            self.action_penalty_weight = 0.0
        elif self.reward_preset == "h_only_straight_walk_normalized_fullscale":
            self.fall_penalty = -2500.0
            self.survival_reward = 0.5
            self.forward_progress_reward_gain = 12000.0
            self.forward_speed_reward_gain = 0.0
            self.roll_pitch_rate_penalty_gain = 1.0
            self.side_roll_penalty_gain = 6.0
            self.action_penalty_weight = 0.4
        else:
            self.fall_penalty = -500.0
            self.survival_reward = 0.5
            self.forward_progress_reward_gain = 2000.0
            self.forward_speed_reward_gain = 5.0
            self.roll_pitch_rate_penalty_gain = 2.0
            self.side_roll_penalty_gain = 6.0
            self.action_penalty_weight = 0.0
        normalized_presets = {
            "h_only_straight_walk_survival_2_action_penalty",
            "h_only_straight_walk_normalized_fullscale",
        }
        self.normalize_roll_pitch_rate = self.reward_preset in normalized_presets
        self.normalize_action_penalty = self.reward_preset in normalized_presets
        self.normalize_side_roll = self.reward_preset in normalized_presets
        self.normalized_reward_scale = (
            10000.0 if self.reward_preset == "h_only_straight_walk_normalized_fullscale" else 1.0
        )
        if episode_seconds is None:
            self.max_steps = 10000
        else:
            episode_seconds = float(episode_seconds)
            if episode_seconds <= 0.0:
                raise ValueError("episode_seconds must be > 0")
            self.max_steps = max(1, int(round(episode_seconds / self.dt)))
        self.reset_settle_time = float(reset_settle_time)
        self.current_step = 0
        self.roll_pitch_rate_sum = 0.0
        self.prev_roll_pitch_rate_average = 0.0
        self.action_penalty_sum = 0.0
        self.prev_action_penalty_average = 0.0
        self.side_roll_sum = 0.0
        self.prev_side_roll_average = 0.0
        self.y_start = 0.0
        self.k_extension_states = self.initial_prismatic_targets[0:4].copy()
        self.f_extension_states = self.initial_prismatic_targets[4:8].copy()
        self.h_extension_states = self.initial_prismatic_targets[8:12].copy()
        self.k_extension_velocities = np.zeros(4, dtype=np.float32)
        self.f_extension_velocities = np.zeros(4, dtype=np.float32)
        self.h_extension_velocities = np.zeros(4, dtype=np.float32)
        self.prev_action = self.initial_prismatic_targets.copy()
        self.real_time_start = 0
        self.reward_term_names = [
            "forward_progress",
            "forward_speed",
            "roll_pitch_rate",
            "side_roll",
            "survival",
            "fall",
            "action_magnitude",
        ]
        self.episode_reward_terms = {name: 0.0 for name in self.reward_term_names}

    def _action_to_prismatic_targets(self, action):
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action_01 = 0.5 * (clipped_action + 1.0)
        selected_targets = (
            self.prismatic_min_limits[self.train_prismatic_indices]
            + action_01 * (
                self.prismatic_max_limits[self.train_prismatic_indices]
                - self.prismatic_min_limits[self.train_prismatic_indices]
            )
        )
        prismatic_targets = np.zeros(12, dtype=np.float32)
        prismatic_targets[self.train_prismatic_indices] = selected_targets
        return prismatic_targets

    def _apply_prismatic_ctrl(self):
        for i in range(4):
            self.data.ctrl[self.k_act_ids[i]] = self.k_extension_states[i]
            self.data.ctrl[self.f_act_ids[i]] = self.f_extension_states[i]
            self.data.ctrl[self.h_act_ids[i]] = self.h_extension_states[i]

    def _advance_prismatic_with_acceleration_limit(self, states, velocities, targets):
        """Post-PPO29 trial: rate- and acceleration-limit prismatic commands."""
        physics_dt = self.model.opt.timestep
        errors = targets - states
        braking_speeds = np.sqrt(
            2.0 * self.prismatic_max_acceleration * np.abs(errors)
        )
        desired_velocities = np.sign(errors) * np.minimum(
            self.k_slew_rate, braking_speeds
        )
        max_velocity_delta = self.prismatic_max_acceleration * physics_dt
        velocities += np.clip(
            desired_velocities - velocities,
            -max_velocity_delta,
            max_velocity_delta,
        )

        previous_errors = errors.copy()
        states += velocities * physics_dt
        remaining_errors = targets - states
        crossed_target = previous_errors * remaining_errors <= 0.0
        states[crossed_target] = targets[crossed_target]
        velocities[crossed_target] = 0.0

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
            self.data.ctrl[self.motor_ids[0]] = self.reset_crank_gain * (target_q_m - self.data.qpos[self.crank_joint_qadrs[0]])
            self.data.ctrl[self.motor_ids[3]] = self.reset_crank_gain * (target_q_r_y - self.data.qpos[self.crank_joint_qadrs[3]])
            mujoco.mj_step(self.model, self.data)
            
        self.damping_switched = False
        self.k_extension_states = self.initial_prismatic_targets[0:4].copy()
        self.f_extension_states = self.initial_prismatic_targets[4:8].copy()
        self.h_extension_states = self.initial_prismatic_targets[8:12].copy()
        self.k_extension_velocities.fill(0.0)
        self.f_extension_velocities.fill(0.0)
        self.h_extension_velocities.fill(0.0)
        self.prev_action = self.initial_prismatic_targets.copy()
        self._apply_prismatic_ctrl()
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        self.roll_pitch_rate_sum = 0.0
        self.prev_roll_pitch_rate_average = 0.0
        self.action_penalty_sum = 0.0
        self.prev_action_penalty_average = 0.0
        self.side_roll_sum = 0.0
        self.prev_side_roll_average = 0.0
        self.y_start = float(self.data.qpos[self.plate_qadr + 1])
        self.real_time_start = time.perf_counter()
        self.episode_reward_terms = {name: 0.0 for name in self.reward_term_names}
        return self._get_obs(), {}

    def step(self, action):
        if not self.damping_switched:
            self.set_hinge_damping(0.5) 
            self.damping_switched = True
        
        # Split 12 normalized actions into k/f/h targets with asymmetric min/max limits.
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        prismatic_targets = self._action_to_prismatic_targets(action)
        target_k = prismatic_targets[0:4]
        target_f = prismatic_targets[4:8]
        target_h = prismatic_targets[8:12]

        x_before = self.data.qpos[self.plate_qadr]
        
        for _ in range(self.frame_skip):
            for mid in self.motor_ids: self.data.ctrl[mid] = self.speed_cmd
            
            # Independent slew control for 4 k-links, 4 f-links and 4 h-links
            if self.prismatic_max_acceleration > 0.0:
                self._advance_prismatic_with_acceleration_limit(
                    self.k_extension_states, self.k_extension_velocities, target_k
                )
                self._advance_prismatic_with_acceleration_limit(
                    self.f_extension_states, self.f_extension_velocities, target_f
                )
                self._advance_prismatic_with_acceleration_limit(
                    self.h_extension_states, self.h_extension_velocities, target_h
                )
            else:
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
        
        x_after = self.data.qpos[self.plate_qadr]
        raw_roll_pitch_rate = -(abs(wx) + abs(wy)) * self.roll_pitch_rate_penalty_gain
        if self.normalize_roll_pitch_rate:
            self.roll_pitch_rate_sum += float(raw_roll_pitch_rate)
            roll_pitch_rate_average = self.roll_pitch_rate_sum / float(self.current_step + 1)
            roll_pitch_rate_reward = (
                roll_pitch_rate_average - self.prev_roll_pitch_rate_average
            ) * self.normalized_reward_scale
            self.prev_roll_pitch_rate_average = roll_pitch_rate_average
        else:
            roll_pitch_rate_reward = raw_roll_pitch_rate

        raw_action_penalty = -self.action_penalty_weight * float(np.mean(np.abs(clipped_action)))
        if self.normalize_action_penalty:
            self.action_penalty_sum += float(raw_action_penalty)
            action_penalty_average = self.action_penalty_sum / float(self.current_step + 1)
            action_penalty_reward = (
                action_penalty_average - self.prev_action_penalty_average
            ) * self.normalized_reward_scale
            self.prev_action_penalty_average = action_penalty_average
        else:
            action_penalty_reward = raw_action_penalty

        raw_side_roll = -abs(up_y) * self.side_roll_penalty_gain
        if self.normalize_side_roll:
            self.side_roll_sum += float(raw_side_roll)
            side_roll_average = self.side_roll_sum / float(self.current_step + 1)
            side_roll_reward = (
                side_roll_average - self.prev_side_roll_average
            ) * self.normalized_reward_scale
            self.prev_side_roll_average = side_roll_average
        else:
            side_roll_reward = raw_side_roll

        reward_terms = {
            "forward_progress": (x_after - x_before) * self.forward_progress_reward_gain,
            "forward_speed": self.forward_speed_reward_gain * float(vx),
            "roll_pitch_rate": roll_pitch_rate_reward,
            "side_roll": side_roll_reward,
            "survival": self.survival_reward,
            "fall": 0.0,
            "action_magnitude": action_penalty_reward,
        }
        reward_terms = {name: float(value) for name, value in reward_terms.items()}
        self.prev_action = prismatic_targets.copy()

        terminated = up_z < 0.6 or self.data.qpos[self.plate_qadr+2] < 0.05
        if terminated:
            reward_terms["fall"] = self.fall_penalty

        reward = float(sum(reward_terms.values()))
        for name, value in reward_terms.items():
            self.episode_reward_terms[name] += float(value)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        info = {
            "reward_terms": reward_terms,
        }
        if terminated or truncated:
            info["episode_reward_terms"] = self.episode_reward_terms.copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer: self.viewer.close()
