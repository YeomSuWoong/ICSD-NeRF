# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def get_ray_directions_lean(i, j, focal, center):
    """
    get_ray_directions but returns only relevant rays
    Inputs:
        i, j: (B) pixel coordinates
        focal: (2), focal length
    Outputs:
        directions: (B, 3), the direction of the rays in camera coordinate
    """
    i, j = i.float() + 0.5, j.float() + 0.5
    directions = torch.stack(
        [(i - center[0]) / focal[0], -(j - center[1]) / focal[1], -torch.ones_like(i)],
        -1,
    )  # (B, 3)

    return directions


def get_rays_lean(directions, c2w):
    """
    get_rays but returns only relevant rays
    Inputs:
        directions: (B, 3) ray directions in camera coordinate
        c2w: (B, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (B, 3), the origin of the rays in world coordinate
        rays_d: (B, 3), the direction of the rays in world coordinate
    """
    rays_d = torch.bmm(directions[:, None, :], torch.transpose(c2w[:, :3, :3], 1, 2))[
        :, 0, :
    ]  # (B, 3)
    rays_o = c2w[:, :3, 3]  # (B, 3)

    return rays_o, rays_d


def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate (OpenGL convention).
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    i, j = i + 0.5, j + 0.5
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the direction of the rays in world coordinate
    """
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Convert rays to normalized device coordinates (forward-facing scenes)."""
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal[0])) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal[1])) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal[0]))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal[1]))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
