"""DART-style action noise injection for motion-planning demo generation.

Laskey et al. 2017 ("DART: Noise Injection for Robust Imitation Learning", CoRL).

The motion planner produces a clean action at each step. We execute a noised
version to diversify visited states, but record the clean action as the BC
training label. This reduces covariate shift without requiring an interactive
expert.
"""

from __future__ import annotations

import h5py
import numpy as np
import sapien
from transforms3d.quaternions import quat2axangle


class DartNoise:
    """Per-step additive Gaussian noise on arm dims, scaled by the action delta.

    ``std_i = scale * |a_t[i] - a_{t-1}[i]|``, clipped to ``[-max_noise, max_noise]``.
    The last ``gripper_dims`` entries of the action vector are left untouched.
    """

    def __init__(
        self,
        scale: float = 0.1,
        max_noise: float = 0.05,
        gripper_dims: int = 1,
        seed: int | None = None,
    ) -> None:
        self.scale = scale
        self.max_noise = max_noise
        self.gripper_dims = gripper_dims
        self.rng = np.random.default_rng(seed)
        self._prev_arm: np.ndarray | None = None
        self.clean_log: list[np.ndarray] = []

    def reset(self) -> None:
        self._prev_arm = None
        self.clean_log = []

    def __call__(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        self.clean_log.append(action.copy())

        if self.gripper_dims > 0:
            arm = action[: -self.gripper_dims]
            tail = action[-self.gripper_dims :]
        else:
            arm = action
            tail = np.array([], dtype=np.float32)

        delta = np.zeros_like(arm) if self._prev_arm is None else arm - self._prev_arm
        self._prev_arm = arm.copy()

        std = self.scale * np.abs(delta)
        eps = self.rng.normal(0.0, 1.0, size=arm.shape).astype(np.float32) * std.astype(np.float32)
        eps = np.clip(eps, -self.max_noise, self.max_noise)
        return np.concatenate([arm + eps, tail]).astype(np.float32)


def wrap_env_step(env, noise: DartNoise):
    """Monkey-patch ``env.step`` to noise actions and log the clean action.

    Returns the original ``step`` callable so the caller can restore it.
    """
    original_step = env.step

    def noised_step(action):
        return original_step(noise(action))

    env.step = noised_step
    return original_step


def rewrite_actions_to_clean(env, noise: DartNoise) -> None:
    """Post-``flush_trajectory`` swap: h5 ``actions`` -> clean, noised -> ``actions_executed``.

    Must be called after ``env.flush_trajectory(save=True)`` while ``env._h5_file``
    is still open. If ``noise.clean_log`` is empty (e.g. failed MP with zero steps),
    this is a no-op.
    """
    if not noise.clean_log:
        return
    traj_id = f"traj_{env._episode_id}"
    if traj_id not in env._h5_file:
        return
    group: h5py.Group = env._h5_file[traj_id]
    clean = np.stack(noise.clean_log, axis=0).astype(np.float32)

    if "actions" in group:
        executed = group["actions"][...]
        if executed.shape != clean.shape:
            raise ValueError(
                f"DART rewrite shape mismatch for {traj_id}: "
                f"executed={executed.shape} clean={clean.shape}"
            )
        del group["actions"]
        group.create_dataset("actions_executed", data=executed, dtype=np.float32)
    group.create_dataset("actions", data=clean, dtype=np.float32)

    group.attrs["dart_noise_scale"] = float(noise.scale)
    group.attrs["dart_noise_max"] = float(noise.max_noise)
    group.attrs["dart_noise_gripper_dims"] = int(noise.gripper_dims)


def _compact_axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(q)
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


class PandaDeltaPoseDartNoise(DartNoise):
    """Panda DART noise + inline FK-based delta-pose label synthesis.

    Drop-in replacement for ``DartNoise`` when the *env is in pd_joint_pos* but
    you want the h5 actions written as ``pd_ee_delta_pose``. Computes per-step
    delta-pose labels for both clean and noised arm targets via pinocchio FK,
    matching Panda's default ``pd_ee_delta_pose`` config (pos/rot bound 0.1,
    ``root_translation:root_aligned_body_rotation`` frame, ``use_target=False``,
    ``normalize_action=True``).

    Must be bound to the env *after* env creation and *before* the first step
    via ``bind(env)``. Requires the Panda robot (``panda``/``panda_wristcam``).
    """

    ARM_NDOF = 7
    EE_LINK_NAME = "panda_hand_tcp"
    POS_BOUND = 0.1
    ROT_BOUND = 0.1

    def __init__(
        self,
        scale: float = 0.1,
        max_noise: float = 0.05,
        gripper_dims: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(scale, max_noise, gripper_dims, seed)
        self._articulation = None
        self._pin_model = None
        self._ee_link_idx: int | None = None
        self._fk_planner = None
        self.delta_clean_log: list[np.ndarray] = []
        self.delta_executed_log: list[np.ndarray] = []

    def bind(self, env) -> None:
        import mplib

        art = env.unwrapped.agent.robot
        self._articulation = art
        urdf_path = env.unwrapped.agent.urdf_path
        link_names = [lk.get_name() for lk in art.get_links()]
        joint_names = [j.get_name() for j in art.get_active_joints()]
        self._fk_planner = mplib.Planner(
            urdf=urdf_path,
            srdf=urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.EE_LINK_NAME,
        )
        self._pin_model = self._fk_planner.pinocchio_model
        self._ee_link_idx = self._fk_planner.link_name_2_idx[self.EE_LINK_NAME]

    def reset(self) -> None:
        super().reset()
        self.delta_clean_log = []
        self.delta_executed_log = []

    def _fk_ee(self, full_qpos: np.ndarray) -> sapien.Pose:
        self._pin_model.compute_forward_kinematics(full_qpos)
        arr = self._pin_model.get_link_pose(self._ee_link_idx)
        return sapien.Pose(p=arr[:3], q=arr[3:])

    def _qpos_target_to_delta_action(
        self, current_full_qpos: np.ndarray, target_arm_qpos: np.ndarray
    ) -> np.ndarray:
        full_target = current_full_qpos.copy()
        full_target[: self.ARM_NDOF] = target_arm_qpos
        cur_ee = self._fk_ee(current_full_qpos)
        tgt_ee = self._fk_ee(full_target)
        delta_p = tgt_ee.p - cur_ee.p
        delta_q = (cur_ee * tgt_ee.inv()).q
        delta_rot = _compact_axis_angle_from_quat(delta_q)
        scaled_p = np.clip(delta_p / self.POS_BOUND, -1.0, 1.0)
        scaled_r = delta_rot / self.ROT_BOUND
        rn = float(np.linalg.norm(scaled_r))
        if rn > 1.0:
            scaled_r = scaled_r / rn
        return np.concatenate([scaled_p, scaled_r]).astype(np.float32)

    def __call__(self, action) -> np.ndarray:
        assert self._articulation is not None, "call bind(env) before stepping"
        action = np.asarray(action, dtype=np.float32)
        clean_full = action.copy()
        self.clean_log.append(clean_full.copy())

        arm = clean_full[: -self.gripper_dims]
        tail = clean_full[-self.gripper_dims :]

        delta = np.zeros_like(arm) if self._prev_arm is None else arm - self._prev_arm
        self._prev_arm = arm.copy()
        std = self.scale * np.abs(delta)
        eps = self.rng.normal(0.0, 1.0, size=arm.shape).astype(np.float32) * std.astype(np.float32)
        eps = np.clip(eps, -self.max_noise, self.max_noise)
        executed_arm = arm + eps
        executed_full = np.concatenate([executed_arm, tail]).astype(np.float32)

        current_qpos = self._articulation.get_qpos().cpu().numpy()[0].astype(np.float64)
        delta_clean_arm = self._qpos_target_to_delta_action(current_qpos, arm.astype(np.float64))
        delta_exec_arm = self._qpos_target_to_delta_action(current_qpos, executed_arm.astype(np.float64))
        self.delta_clean_log.append(np.concatenate([delta_clean_arm, tail]).astype(np.float32))
        self.delta_executed_log.append(np.concatenate([delta_exec_arm, tail]).astype(np.float32))

        return executed_full


def rewrite_actions_to_delta_pose(env, noise: PandaDeltaPoseDartNoise) -> None:
    """Post-``flush_trajectory`` swap: write delta-pose labels to h5.

    Overwrites ``actions`` with clean delta-pose labels, writes noised
    delta-pose as ``actions_executed``, and keeps the original joint-pos
    streams under ``actions_joint_clean`` / ``actions_joint_executed`` for
    debugging.
    """
    if not noise.delta_clean_log:
        return
    traj_id = f"traj_{env._episode_id}"
    if traj_id not in env._h5_file:
        return
    group: h5py.Group = env._h5_file[traj_id]

    delta_clean = np.stack(noise.delta_clean_log, axis=0).astype(np.float32)
    delta_exec = np.stack(noise.delta_executed_log, axis=0).astype(np.float32)
    joint_clean = np.stack(noise.clean_log, axis=0).astype(np.float32)

    if "actions" in group:
        joint_executed = group["actions"][...]
        if joint_executed.shape != joint_clean.shape:
            raise ValueError(
                f"DART rewrite shape mismatch for {traj_id}: "
                f"executed={joint_executed.shape} clean={joint_clean.shape}"
            )
        del group["actions"]
        group.create_dataset("actions_joint_executed", data=joint_executed, dtype=np.float32)
    group.create_dataset("actions_joint_clean", data=joint_clean, dtype=np.float32)
    group.create_dataset("actions_executed", data=delta_exec, dtype=np.float32)
    group.create_dataset("actions", data=delta_clean, dtype=np.float32)

    group.attrs["dart_noise_scale"] = float(noise.scale)
    group.attrs["dart_noise_max"] = float(noise.max_noise)
    group.attrs["dart_noise_gripper_dims"] = int(noise.gripper_dims)
    group.attrs["target_control_mode"] = "pd_ee_delta_pose"
