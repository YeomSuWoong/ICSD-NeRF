# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def pose_to_mtx(pose):
    """(N, 9) pose parameters -> (N, 3, 4) camera-to-world matrices.

    The first 6 values are a continuous 6D rotation representation (two columns
    orthonormalized by Gram-Schmidt); the last 3 values are the translation.
    """
    b1 = pose[..., 0:3]
    b1 = b1 / torch.norm(b1, dim=-1)[:, None]
    b2 = pose[..., 3:6] - torch.sum(b1 * pose[..., 3:6], dim=-1)[:, None] * b1
    b2 = b2 / torch.norm(b2, dim=-1)[:, None]
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3, pose[..., 6:9]], dim=-1)
