# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys

import imageio
import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm.auto import tqdm

from dataLoader.ray_utils import get_ray_directions_blender, get_rays, ndc_rays
from flow_viz import flow_to_image
from utils import rgb_lpips, rgb_ssim, visualize_depth


def sampleXYZ(tensorf, rays, N_samples, ray_type="ndc", is_train=False):
    if ray_type == "ndc":
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_ndc(
            rays[:, :3], rays[:, 3:6], is_train=is_train, N_samples=N_samples
        )
    elif ray_type == "contract":
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays[:, :3], rays[:, 3:6], is_train=is_train, N_samples=N_samples
        )
    z_vals = torch.tile(z_vals, (xyz_sampled.shape[0], 1))
    return xyz_sampled, z_vals, ray_valid


def raw2outputs(
    rgb_s,
    sigma_s,
    rgb_d,
    sigma_d,
    dists,
    blending,
    z_vals,
    rays_chunk,
    is_train=False,
    ray_type="ndc",
):
    """Composite the static and dynamic fields along each ray.

    `blending` is the rigidity predicted by the static field: a sample
    contributes with alpha_s * blending (static) and alpha_d * (1 - blending)
    (dynamic). The static-only and dynamic-only renderings are returned as well.
    """

    def raw2alpha(sigma, dist):
        return 1.0 - torch.exp(-sigma * dist)

    blending = blending.squeeze(-1)
    alpha_d = raw2alpha(sigma_d, dists)  # [N_rays, N_samples]
    alpha_s = raw2alpha(sigma_s, dists)  # [N_rays, N_samples]

    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device),
                1.0 - alpha_d + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_s.shape[0], 1)).to(alpha_s.device),
                1.0 - alpha_s + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_full = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device),
                (1.0 - alpha_d * (1.0 - blending)) * (1.0 - alpha_s * blending) + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]

    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_d = weights_d / (torch.sum(weights_d, -1, keepdim=True) + 1e-10)
    weights_full = (alpha_d * (1 - blending) + alpha_s * blending) * T_full

    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * (1 - blending))[..., None] * rgb_d
        + (T_full * alpha_s * blending)[..., None] * rgb_s,
        -2,
    )

    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    # random white background during training
    if is_train and torch.rand((1,)) < 0.5:
        rgb_map_d = rgb_map_d + (1.0 - acc_map_d[..., None])
        rgb_map_s = rgb_map_s + (1.0 - acc_map_s[..., None])
        rgb_map_full = rgb_map_full + torch.relu(1.0 - acc_map_full[..., None])

    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)
    if ray_type == "ndc":
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
    elif ray_type == "contract":
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * 256.0
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * 256.0
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * 256.0

    rgb_map_d = rgb_map_d.clamp(0, 1)
    rgb_map_s = rgb_map_s.clamp(0, 1)
    rgb_map_full = rgb_map_full.clamp(0, 1)

    # dynamicness: accumulated contribution of the dynamic field
    dynamicness_map = torch.sum(weights_full * (1 - blending), -1)

    return edict(
        rgb_map=rgb_map_full,
        depth_map=depth_map_full,
        acc_map=acc_map_full,
        weights=weights_full,
        rgb_map_s=rgb_map_s,
        depth_map_s=depth_map_s,
        acc_map_s=acc_map_s,
        weights_s=weights_s,
        rgb_map_d=rgb_map_d,
        depth_map_d=depth_map_d,
        acc_map_d=acc_map_d,
        weights_d=weights_d,
        dynamicness_map=dynamicness_map,
    )


def render_composite(
    tensorf_static,
    tensorf_dynamic_g,
    tensorf_dynamic_c,
    rays,
    ts,
    N_samples,
    ray_type,
    is_train,
    detach_static=False,
):
    """Render rays with the static, dynamic-geometry and dynamic-color fields
    and composite them (see `raw2outputs`).

    detach_static: stop the gradients of the static color/density (the
        rigidity/blending still receives gradients).
    """
    xyz_sampled, z_vals, ray_valid = sampleXYZ(
        tensorf_dynamic_g, rays, N_samples=N_samples, ray_type=ray_type, is_train=is_train
    )
    blending, _, rgb_s, sigma_s = tensorf_static(rays, ts, xyz_sampled, z_vals, ray_valid)
    sigma_d, dists = tensorf_dynamic_g(rays, ts, xyz_sampled, z_vals, ray_valid)
    rgb_d = tensorf_dynamic_c(rays, ts, xyz_sampled, z_vals, ray_valid)
    if detach_static:
        rgb_s, sigma_s = rgb_s.detach(), sigma_s.detach()

    out = raw2outputs(
        rgb_s,
        sigma_s,
        rgb_d,
        sigma_d,
        dists,
        blending,
        z_vals,
        rays,
        is_train=is_train,
        ray_type=ray_type,
    )
    out.pts = xyz_sampled
    out.z_vals = z_vals
    return out


def rays_for_view(c2w, H, W, focal, ray_type):
    directions = get_ray_directions_blender(H, W, [focal, focal]).to(c2w.device)
    rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
    if ray_type == "ndc":
        rays_o, rays_d = ndc_rays(H, W, [focal, focal], 1.0, rays_o, rays_d)
    return torch.cat([rays_o, rays_d], 1)  # (h*w, 6)


@torch.no_grad()
def render_image(
    tensorf_static,
    tensorf_dynamic_g,
    tensorf_dynamic_c,
    rays,
    ts,
    H,
    W,
    chunk,
    N_samples,
    ray_type,
    device,
):
    """Render a full image; returns composite / static / dynamic RGB and depth."""
    keys = ["rgb_map", "depth_map", "rgb_map_s", "depth_map_s", "rgb_map_d", "depth_map_d", "dynamicness_map"]
    results = {k: [] for k in keys}
    for chunk_idx in range(rays.shape[0] // chunk + int(rays.shape[0] % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        out = render_composite(
            tensorf_static,
            tensorf_dynamic_g,
            tensorf_dynamic_c,
            rays_chunk,
            ts_chunk,
            N_samples,
            ray_type,
            is_train=False,
        )
        for k in keys:
            results[k].append(out[k])
    results = edict({k: torch.cat(v) for k, v in results.items()})

    results.rgb_map = results.rgb_map.clamp(0.0, 1.0).reshape(H, W, 3).cpu()
    results.rgb_map_s = results.rgb_map_s.clamp(0.0, 1.0).reshape(H, W, 3).cpu()
    results.rgb_map_d = results.rgb_map_d.clamp(0.0, 1.0).reshape(H, W, 3).cpu()
    results.blending_map = results.dynamicness_map.clamp(0.0, 1.0).reshape(H, W).cpu()
    results.depth_map = results.depth_map.reshape(H, W).cpu()
    results.depth_map_s = results.depth_map_s.reshape(H, W).cpu()
    results.depth_map_d = results.depth_map_d.reshape(H, W).cpu()
    return results


def to_uint8(img):
    return (img.numpy() * 255).astype("uint8")


def save_frames_and_videos(savePath, idx, res, frames):
    """Save composite / static / dynamic frames of one image and collect them for the videos."""
    rgb, rgb_s, rgb_d, blending = (
        to_uint8(res.rgb_map),
        to_uint8(res.rgb_map_s),
        to_uint8(res.rgb_map_d),
        to_uint8(res.blending_map),
    )
    frames["full"].append(rgb)
    frames["static"].append(rgb_s)
    frames["dynamic"].append(rgb_d)
    imageio.imwrite(f"{savePath}/{idx:03d}.png", rgb, format="png")
    imageio.imwrite(f"{savePath}_static/{idx:03d}.png", rgb_s, format="png")
    imageio.imwrite(f"{savePath}_dynamic/{idx:03d}.png", rgb_d, format="png")
    imageio.imwrite(f"{savePath}_dynamic/{idx:03d}_blending.png", blending, format="png")


def write_videos(savePath, frames):
    for key, suffix in [("full", ""), ("static", "_static"), ("dynamic", "_dynamic")]:
        imageio.mimwrite(
            f"{savePath}{suffix}/video.mp4",
            np.stack(frames[key]),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )


def make_output_dirs(savePath):
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "_static/rgbd", exist_ok=True)
    os.makedirs(savePath + "_dynamic/rgbd", exist_ok=True)


@torch.no_grad()
def evaluation(
    test_dataset,
    poses_mtx,
    focal,
    tensorf_static,
    tensorf_dynamic_c,
    tensorf_dynamic_g,
    savePath,
    N_samples=-1,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
):
    """Render every training view (all frames), save images / videos and report
    PSNR (+ SSIM, LPIPS). Returns (PSNRs, near_fars, depth maps)."""
    PSNRs, ssims, l_alex, l_vgg = [], [], [], []
    near_fars, depth_maps = [], []
    frames = {"full": [], "static": [], "dynamic": []}
    make_output_dirs(savePath)

    W, H = test_dataset.img_wh
    for idx in tqdm(range(poses_mtx.shape[0]), file=sys.stdout):
        rays = rays_for_view(poses_mtx[idx], H, W, focal, ray_type)
        ts = test_dataset.all_ts[idx].view(-1)
        res = render_image(
            tensorf_static,
            tensorf_dynamic_g,
            tensorf_dynamic_c,
            rays,
            ts,
            H,
            W,
            chunk=512,
            N_samples=N_samples,
            ray_type=ray_type,
            device=device,
        )

        if ray_type == "contract":
            near_fars.append(
                (
                    torch.quantile(res.depth_map_s, 0.01).item(),
                    torch.quantile(res.depth_map_s, 0.99).item(),
                )
            )
            res.depth_map_s = -1.0 / (res.depth_map_s + 1e-6)
            res.depth_map_d = -1.0 / (res.depth_map_d + 1e-6)
            res.depth_map = -1.0 / (res.depth_map + 1e-6)
        else:
            near_fars.append(
                (
                    torch.quantile(1.0 / (res.depth_map_s + 1e-6), 0.01).item(),
                    torch.quantile(1.0 / (res.depth_map_s + 1e-6), 0.99).item(),
                )
            )

        np.save(f"{savePath}_static/rgbd/{idx:03d}.npy", res.depth_map_s)
        np.save(f"{savePath}/rgbd/{idx:03d}.npy", res.depth_map)

        gt_rgb = test_dataset.all_rgbs[idx].view(H, W, 3)
        loss = torch.mean((res.rgb_map - gt_rgb) ** 2)
        PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
        if compute_extra_metrics:
            ssims.append(rgb_ssim(res.rgb_map, gt_rgb, 1))
            l_alex.append(
                rgb_lpips(gt_rgb.numpy(), res.rgb_map.numpy(), "alex", tensorf_dynamic_c.device)
            )
            l_vgg.append(
                rgb_lpips(gt_rgb.numpy(), res.rgb_map.numpy(), "vgg", tensorf_dynamic_c.device)
            )

        depth_maps.append(res.depth_map)
        save_frames_and_videos(savePath, idx, res, frames)

    write_videos(savePath, frames)

    psnr = np.mean(np.asarray(PSNRs))
    if compute_extra_metrics:
        metrics = [psnr, np.mean(ssims), np.mean(l_alex), np.mean(l_vgg)]
    else:
        metrics = [psnr]
    np.savetxt(f"{savePath}/mean.txt", np.asarray(metrics))

    return PSNRs, near_fars, depth_maps


@torch.no_grad()
def evaluation_path(
    test_dataset,
    tensorf_static,
    tensorf_dynamic_c,
    tensorf_dynamic_g,
    c2ws,
    focals,
    savePath,
    N_samples=-1,
    ray_type="ndc",
    change_time="change",
    device="cuda",
):
    """Render a camera path (novel views / novel times).

    change_time: "change" renders frame i at time i (time follows the path),
        a float in [-1, 1] renders every view at that fixed time.
    Returns the depth maps.
    """
    depth_maps = []
    frames = {"full": [], "static": [], "dynamic": []}
    make_output_dirs(savePath)

    W, H = test_dataset.img_wh
    for idx, c2w in tqdm(enumerate(c2ws)):
        rays = rays_for_view(torch.FloatTensor(c2w), H, W, focals[idx], ray_type)
        if change_time == "change":
            t = idx / (len(c2ws) - 1) * 2.0 - 1.0
        else:
            t = change_time
        ts = t * torch.ones(W * H)

        res = render_image(
            tensorf_static,
            tensorf_dynamic_g,
            tensorf_dynamic_c,
            rays,
            ts,
            H,
            W,
            chunk=8192,
            N_samples=N_samples,
            ray_type=ray_type,
            device=device,
        )
        if ray_type == "contract":
            res.depth_map = -1.0 / (res.depth_map + 1e-6)

        depth_maps.append(res.depth_map)
        save_frames_and_videos(savePath, idx, res, frames)

    write_videos(savePath, frames)
    return depth_maps


@torch.no_grad()
def render(
    test_dataset,
    poses_mtx,
    focal,
    tensorf_static,
    tensorf_dynamic_c,
    tensorf_dynamic_g,
    N_vis=5,
    N_samples=-1,
    ray_type="ndc",
    device="cuda",
):
    """Render test views for TensorBoard, including induced flows and the
    deformation of the dynamic geometry field."""
    vis = edict(
        rgb_maps=[],
        depth_maps=[],
        blending_maps=[],
        gt_rgbs=[],
        induced_flow_f=[],
        induced_flow_b=[],
        induced_flow_s_f=[],
        induced_flow_s_b=[],
        delta_xyz=[],
        rgb_maps_s=[],
        depth_maps_s=[],
        rgb_maps_d=[],
        depth_maps_d=[],
        monodepth=[],
    )

    W, H = test_dataset.img_wh
    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).view(-1, 2).to(device)

    n_views = poses_mtx.shape[0]
    img_eval_interval = 1 if N_vis < 0 else max(n_views // N_vis, 1)
    for idx in range(0, n_views, img_eval_interval):
        rays = rays_for_view(poses_mtx[idx], H, W, focal, ray_type)
        ts = test_dataset.all_ts[idx].view(-1)
        pose_f = poses_mtx[min(idx + 1, n_views - 1), :3, :4]
        pose_b = poses_mtx[max(idx - 1, 0), :3, :4]

        keys = [
            "rgb_map", "rgb_map_s", "rgb_map_d", "dynamicness_map", "depth_map",
            "depth_map_s", "depth_map_d", "flow_f", "flow_b", "flow_s_f", "flow_s_b",
            "weights_d", "delta_xyz",
        ]
        acc = {k: [] for k in keys}
        chunk = 1024
        for chunk_idx in range(rays.shape[0] // chunk + int(rays.shape[0] % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            grid_chunk = grid[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            out = render_composite(
                tensorf_static,
                tensorf_dynamic_g,
                tensorf_dynamic_c,
                rays_chunk,
                ts_chunk,
                N_samples,
                ray_type,
                is_train=False,
            )
            pts = out.pts
            delta_xyz = (
                tensorf_dynamic_g.warp_coordinate(
                    pts, tensorf_dynamic_g.tile_time(ts_chunk, pts.shape[1])
                )
                - pts
            )
            # flow induced by the dynamic field (scene flow) and by the static field (camera motion)
            scene_flow_f, scene_flow_b = tensorf_dynamic_g.get_forward_backward_scene_flow(
                pts, ts_chunk
            )
            n = out.weights_d.shape[0]
            for key, pose, weights, pts_moved in [
                ("flow_f", pose_f, out.weights_d, pts + scene_flow_f),
                ("flow_b", pose_b, out.weights_d, pts + scene_flow_b),
                ("flow_s_f", pose_f, out.weights_s, pts),
                ("flow_s_b", pose_b, out.weights_s, pts),
            ]:
                flow, _ = induce_flow(
                    H,
                    W,
                    focal,
                    torch.tile(pose[None], (n, 1, 1)),
                    weights,
                    pts_moved,
                    grid_chunk,
                    rays_chunk,
                    ray_type=ray_type,
                )
                acc[key].append(flow)
            for key in ["rgb_map", "rgb_map_s", "rgb_map_d", "dynamicness_map", "depth_map", "depth_map_s", "depth_map_d", "weights_d"]:
                acc[key].append(out[key])
            acc["delta_xyz"].append(delta_xyz)
        acc = edict({k: torch.cat(v) for k, v in acc.items()})

        depth_map = acc.depth_map.reshape(H, W)
        depth_map_s = acc.depth_map_s.reshape(H, W)
        depth_map_d = acc.depth_map_d.reshape(H, W)
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        vis.rgb_maps.append(acc.rgb_map.clamp(0.0, 1.0).reshape(H, W, 3))
        vis.rgb_maps_s.append(acc.rgb_map_s.clamp(0.0, 1.0).reshape(H, W, 3))
        vis.rgb_maps_d.append(acc.rgb_map_d.clamp(0.0, 1.0).reshape(H, W, 3))
        vis.depth_maps.append(depth_map)
        vis.depth_maps_s.append(depth_map_s)
        vis.depth_maps_d.append(depth_map_d)
        vis.blending_maps.append(
            acc.dynamicness_map.clamp(0.0, 1.0).reshape(H, W)[None].repeat(3, 1, 1)
        )
        vis.gt_rgbs.append(test_dataset.all_rgbs[idx].view(H, W, 3))
        for key in ["flow_f", "flow_b", "flow_s_f", "flow_s_b"]:
            vis["induced_" + key].append(
                torch.from_numpy(flow_to_image(acc[key].view(H, W, 2).detach().cpu().numpy()))
            )
        delta_xyz_sum = torch.sum(acc.weights_d[..., None] * acc.delta_xyz, 1).view(H, W, 3)
        vis.delta_xyz.append(
            ((delta_xyz_sum / torch.max(torch.abs(delta_xyz_sum))) + 1.0) / 2.0
        )

    all_depth = torch.stack(vis.depth_maps + vis.depth_maps_s + vis.depth_maps_d)
    depth_map_min = torch.min(all_depth).item()
    depth_map_max = torch.max(all_depth).item()
    for key in ["depth_maps", "depth_maps_s", "depth_maps_d"]:
        vis[key] = [
            visualize_depth(
                torch.clamp(d, min=depth_map_min, max=depth_map_max),
                (depth_map_min, depth_map_max),
            )[0]
            for d in vis[key]
        ]

    for i in range(test_dataset.all_disps.shape[0]):
        vis.monodepth.append(visualize_depth(test_dataset.all_disps[i])[0])

    return vis


def NDC2world(pts, H, W, f):
    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1.0, max=1 - 1e-6) - 1)
    pts_x = -pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = -pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def world2NDC(pts_world, H, W, f):
    o0 = -1.0 / (W / (2.0 * f)) * pts_world[..., 0:1] / pts_world[..., 2:]
    o1 = -1.0 / (H / (2.0 * f)) * pts_world[..., 1:2] / pts_world[..., 2:]
    o2 = 1.0 + 2.0 * 1 / pts_world[..., 2:]
    pts = torch.cat([o0, o1, o2], -1)

    return pts


def contract2world(pts_contract):
    pts_norm, _ = torch.max(torch.abs(pts_contract.clone()), dim=-1)
    contract_mask = pts_norm > 1.0
    scale = -1 / (pts_norm - 2)
    pts_world = pts_contract
    pts_world[~contract_mask] = pts_contract[~contract_mask]
    pts_world[contract_mask] = (
        pts_contract[contract_mask]
        / (pts_norm[contract_mask][:, None])
        * scale[contract_mask][:, None]
    )
    return pts_world


def render_3d_point(H, W, f, c2w, weights, pts, rays, ray_type="ndc"):
    """Render 3D position along each ray and project it to the image plane."""
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # Rendered 3D position in NDC / contracted coordinate
    acc_map = torch.sum(weights, -1)[:, None]
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)
    if ray_type == "ndc":
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * (rays[:, :3] + rays[:, 3:])
    elif ray_type == "contract":
        farest_pts = rays[:, :3] + rays[:, 3:] * 256.0
        # convert to contract domain
        farest_pts_norm, _ = torch.max(torch.abs(farest_pts.clone()), dim=-1)
        contract_mask = farest_pts_norm > 1.0
        farest_pts[contract_mask] = (2 - 1 / farest_pts_norm[contract_mask])[
            ..., None
        ] * (farest_pts[contract_mask] / farest_pts_norm[contract_mask][..., None])
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * farest_pts

    # NDC coordinate to world coordinate
    if ray_type == "ndc":
        pts_map_world = NDC2world(pts_map_NDC, H, W, f)
    elif ray_type == "contract":
        pts_map_world = contract2world(pts_map_NDC)

    # World coordinate to camera coordinate
    pts_map_world = pts_map_world - c2w[..., 3]
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat(
        [
            pts_map_cam[..., 0:1] / (-pts_map_cam[..., 2:]) * f + W * 0.5,
            -pts_map_cam[..., 1:2] / (-pts_map_cam[..., 2:]) * f + H * 0.5,
        ],
        -1,
    )

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, pts_map_cam_NDC[:, 2:]


def induce_flow(
    H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d, rays, ray_type="ndc"
):
    """Project the rendered 3D points into the neighbor frame; returns the
    induced 2D flow and the induced disparity."""
    pts_2d_neighbor, induced_disp = render_3d_point(
        H, W, focal, pose_neighbor, weights, pts_3d_neighbor, rays, ray_type
    )
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow, induced_disp
