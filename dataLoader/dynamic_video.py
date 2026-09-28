# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Monocular video datasets.

Expected layout of a scene folder (produced by the scripts in `scripts/`):

    <scene>/
        images/                 RGB frames
        <disp_dir>/{idx}.npy    monocular disparity (DPT)
        flow/{idx}_fwd.npz      forward optical flow + consistency mask (RAFT)
        flow/{idx}_bwd.npz      backward optical flow + consistency mask (RAFT)
        <mask_dir>/*.png        motion masks (`--use_foreground_mask`)
        poses_bounds.npy        (optional) LLFF-format camera poses

`{idx}` is zero-padded to `index_digits` digits.
"""

import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    center = poses[..., 3].mean(0)  # (3)
    z = normalize(poses[..., 2].mean(0))  # (3)
    y_ = poses[..., 1].mean(0)  # (3)
    x = normalize(np.cross(z, y_))  # (3)
    y = np.cross(x, z)  # (3)
    return np.stack([x, y, z, center], 1)  # (3, 4)


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
    """
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = average_poses(poses)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    return poses_centered[:, :3]


def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized


def resize_disp(disp, H_new, W_new):
    return cv2.resize(disp, (W_new, H_new), interpolation=cv2.INTER_LINEAR)


class DynamicVideoDataset(Dataset):
    disp_dir = None
    index_digits = None

    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=False,
        use_foreground_mask="motion_masks",
        with_GT_poses=False,
        ray_type="ndc",
    ):
        """
        All frames are used for both training and evaluation.
        is_stack: return per-frame tensors (N, H, W, C) instead of flattened rays (N*H*W, C).
        """
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.transform = T.ToTensor()
        self.with_GT_poses = with_GT_poses
        self.use_foreground_mask = use_foreground_mask
        self.ray_type = ray_type

        self.read_meta()
        self.white_bg = False

        if ray_type == "contract":
            self.near_far = [0.0, 256]
            self.scene_bbox = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        else:
            self.near_far = [0.0, 1.0]
            self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

    @property
    def num_frames(self):
        return len(self.image_paths)

    def frame_file(self, folder, i, suffix):
        return os.path.join(
            self.root_dir, folder, str(i).zfill(self.index_digits) + suffix
        )

    def read_meta(self):
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "images/*")))
        self.foreground_mask_paths = sorted(
            glob.glob(os.path.join(self.root_dir, self.use_foreground_mask, "*.png"))
        )

        tmp_img = np.array(Image.open(self.image_paths[0]).convert("RGB"))
        H, W = tmp_img.shape[0], tmp_img.shape[1]
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        # Initial focal length when camera parameters are unknown.
        self.focal = [
            (max(H, W) / 2 * np.sqrt(3)) / float(self.downsample),
            (max(H, W) / 2 * np.sqrt(3)) / float(self.downsample),
        ]

        img_list = list(range(self.num_frames))

        if self.with_GT_poses:
            self.load_gt_poses(img_list)

        W, H = self.img_wh
        self.all_rgbs = []
        self.all_ts = []
        self.all_disps = []
        self.all_foreground_masks = []
        self.all_flows_f = []
        self.all_flow_masks_f = []
        self.all_flows_b = []
        self.all_flow_masks_b = []
        for idx, i in enumerate(img_list):
            disp_data = np.load(self.frame_file(self.disp_dir, i, ".npy"))
            disp = torch.from_numpy(resize_disp(disp_data, H, W)).view(-1)
            self.all_disps.append(disp)

            img = Image.open(self.image_paths[i]).convert("RGB")
            if self.downsample != 1.0:
                img = img.resize(tuple(self.img_wh.tolist()), Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            self.all_rgbs += [img.view(3, -1).permute(1, 0)]  # (h*w, 3) RGB

            foreground_mask = Image.open(self.foreground_mask_paths[i]).convert("RGB")
            if self.downsample != 1.0:
                foreground_mask = foreground_mask.resize(
                    tuple(self.img_wh.tolist()), Image.BILINEAR
                )
            foreground_mask = self.transform(foreground_mask)  # (3, h, w)
            self.all_foreground_masks += [foreground_mask.view(3, -1).permute(1, 0)]

            # forward flow (none for the last frame)
            if idx == len(img_list) - 1:
                fwd_flow, fwd_mask = torch.zeros(H, W, 2).view(-1, 2), torch.zeros(
                    H, W
                ).view(-1)
            else:
                fwd_flow, fwd_mask = self.load_flow(i, "_fwd.npz", H, W)
            self.all_flows_f.append(fwd_flow)
            self.all_flow_masks_f.append(fwd_mask)

            # backward flow (none for the first frame)
            if idx == 0:
                bwd_flow, bwd_mask = torch.zeros(H, W, 2).view(-1, 2), torch.zeros(
                    H, W
                ).view(-1)
            else:
                bwd_flow, bwd_mask = self.load_flow(i, "_bwd.npz", H, W)
            self.all_flows_b.append(bwd_flow)
            self.all_flow_masks_b.append(bwd_mask)

            # time normalized to [-1, 1]
            self.all_ts += [(idx / (len(img_list) - 1) * 2.0 - 1.0) * torch.ones(H * W)]

        if self.with_GT_poses:
            self.all_poses = torch.stack(self.all_poses, 0)[: len(img_list)]

        if not self.is_stack:
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (N*h*w, 3)
            self.all_ts = torch.cat(self.all_ts, 0)  # (N*h*w)
            self.all_disps = torch.cat(self.all_disps, 0)
            self.all_flows_f = torch.cat(self.all_flows_f, 0)
            self.all_flow_masks_f = torch.cat(self.all_flow_masks_f, 0)
            self.all_flows_b = torch.cat(self.all_flows_b, 0)
            self.all_flow_masks_b = torch.cat(self.all_flow_masks_b, 0)
            self.all_foreground_masks = torch.cat(self.all_foreground_masks, 0)
        else:
            hw = self.img_wh[::-1]
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *hw, 3)
            self.all_ts = torch.stack(self.all_ts, 0).reshape(-1, *hw)
            self.all_disps = torch.stack(self.all_disps, 0).reshape(-1, *hw)
            self.all_flows_f = torch.stack(self.all_flows_f, 0).reshape(-1, *hw, 2)
            self.all_flow_masks_f = torch.stack(self.all_flow_masks_f, 0).reshape(-1, *hw)
            self.all_flows_b = torch.stack(self.all_flows_b, 0).reshape(-1, *hw, 2)
            self.all_flow_masks_b = torch.stack(self.all_flow_masks_b, 0).reshape(-1, *hw)
            self.all_foreground_masks = torch.stack(
                self.all_foreground_masks, 0
            ).reshape(-1, *hw, 3)

    def load_flow(self, i, suffix, H, W):
        data = np.load(self.frame_file("flow", i, suffix))
        flow = torch.from_numpy(resize_flow(data["flow"], H, W)).view(-1, 2)
        mask = torch.from_numpy(
            cv2.resize(np.float32(data["mask"]), (W, H), interpolation=cv2.INTER_NEAREST)
        ).view(-1)
        return flow, mask

    def load_gt_poses(self, img_list):
        poses_bounds = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [focal * self.img_wh[0] / W, focal * self.img_wh[1] / H]

        # Step 2: "down right back" -> "right up back"
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.poses = center_poses(poses)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        if self.ray_type == "ndc":
            scale_factor = self.near_fars.min() * 0.75
            self.near_fars /= scale_factor
        else:
            scale_factor = np.abs(self.poses[..., 3]).max() * 2.0
        self.poses[..., 3] /= scale_factor

        self.all_poses = []
        for i in img_list:
            c2w = torch.FloatTensor(self.poses[i])  # [3, 4]
            c2w[0] = -c2w[0]
            self.all_poses.append(c2w)

    def __len__(self):
        return len(self.all_rgbs)


class NvidiaDataset(DynamicVideoDataset):
    """NVIDIA Dynamic Scenes (preprocessed as in DynamicNeRF)."""

    disp_dir = "disp"
    index_digits = 3


class DavisDataset(DynamicVideoDataset):
    """DAVIS / in-the-wild videos (preprocessed with `scripts/`)."""

    disp_dir = "dpt"
    index_digits = 5
