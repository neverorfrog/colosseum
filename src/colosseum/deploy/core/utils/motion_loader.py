import os
from typing import Optional, Sequence
import numpy as np
import torch
from booster_deploy.utils.isaaclab import math as lab_math


class MotionLoader:
    def __init__(self, motion_file: str, body_names: Sequence[str], *,
                 track_body_names: Optional[Sequence[str]] = None,
                 align_to_first_frame: bool = False, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]

        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device)

        if align_to_first_frame:
            init_root_pos_xy = self._body_pos_w[:1, :1].clone()
            init_root_pos_xy[:, :, 2] = 0.0
            init_root_quat_yaw = lab_math.yaw_quat(self._body_quat_w[:1, :1])
            self._body_pos_w, self._body_quat_w = lab_math.subtract_frame_transforms(
                init_root_pos_xy,
                init_root_quat_yaw.repeat(*self._body_quat_w.shape[:2], 1),
                t02=self._body_pos_w, q02=self._body_quat_w
            )

            q_inv = lab_math.quat_inv(init_root_quat_yaw)
            self._body_lin_vel_w = lab_math.quat_apply(
                q_inv, self._body_lin_vel_w)
            self._body_ang_vel_w = lab_math.quat_apply(
                q_inv, self._body_ang_vel_w)

        self.track_body_names = track_body_names or body_names
        self.body_names = body_names
        self._body_indexes = [
            self.body_names.index(name) for name in self.track_body_names
        ]
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]
