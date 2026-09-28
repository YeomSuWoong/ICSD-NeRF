# Copyright (c) Meta Platforms, Inc. and affiliates.
import datetime
import os
import sys
import time

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_efficient_distloss import flatten_eff_distloss
from tqdm.auto import tqdm

from camera import pose_to_mtx
from dataLoader import dataset_dict
from dataLoader.ray_utils import get_ray_directions_lean, get_rays_lean, ndc_rays
from flow_viz import flow_to_image
from models.tensoRF import (
    DynamicColorTensorVM,
    DynamicGeometryTensorVM,
    StaticTensorVM,
)
from opt import check_hyperparameters, config_parser
from renderer import (
    evaluation,
    evaluation_path,
    induce_flow,
    render,
    render_composite,
    sampleXYZ,
)
from utils import N_to_reso, TVLoss, cal_n_samples, visualize_depth_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


def ids2pixel(W, H, ids):
    """Flattened ray index -> (column, row, frame)."""
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids


def set_num_frames(args, dataset):
    if args.N_voxel_t is None:
        args.N_voxel_t = dataset.num_frames
    assert args.N_voxel_t == dataset.num_frames, "N_voxel_t must equal the number of frames"


# ---------------------------------------------------------------------------
# Rays
# ---------------------------------------------------------------------------
def build_rays(directions, c2w, H, W, focal, ray_type):
    rays_o, rays_d = get_rays_lean(directions, c2w)  # both (B, 3)
    if ray_type == "ndc":
        rays_o, rays_d = ndc_rays(H, W, [focal, focal], 1.0, rays_o, rays_d)
    return torch.cat([rays_o, rays_d], -1).view(-1, 6)


def uv_to_directions(uv, focal, W, H):
    """Camera-space ray directions through (sub-)pixel positions uv (B, 2)."""
    return torch.stack(
        [
            (uv[..., 0] - W / 2) / focal,
            -(uv[..., 1] - H / 2) / focal,
            -torch.ones_like(uv[..., 0]),
        ],
        -1,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def compute_depth_loss(dyn_depth, gt_depth):
    """Scale-and-shift invariant depth loss (median / mean absolute deviation)."""
    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / (s_d + 1e-10)

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / (s_gt + 1e-10)

    return torch.sum((dyn_depth_norm - gt_depth_norm) ** 2)


def monodepth_loss(depth, disp, t_ref, n_frames, ray_type, valid_mask=None):
    """Monocular depth loss, normalized separately for every frame."""
    total, counter = 0.0, 0.0
    for cam_idx in range(n_frames):
        valid = t_ref == cam_idx
        if valid_mask is not None:
            valid = torch.bitwise_and(valid, valid_mask)
        if torch.sum(valid) > 1.0:
            if ray_type == "ndc":
                total += compute_depth_loss(depth[valid], -disp[valid])
            elif ray_type == "contract":
                total += compute_depth_loss(1.0 / (depth[valid] + 1e-6), disp[valid])
            counter += torch.sum(valid)
    return total / counter


def order_loss(depth_d, depth_s, dynamicness, ray_type):
    """In static regions the dynamic field should not be in front of the static field."""
    static_w = 1.0 - dynamicness.detach()
    if ray_type == "ndc":
        diff = (depth_d - depth_s.detach()) ** 2
    elif ray_type == "contract":
        diff = (1.0 / (depth_d + 1e-6) - 1.0 / (depth_s.detach() + 1e-6)) ** 2
    return torch.sum(diff * static_w) / (torch.sum(static_w) + 1e-8)


def skewed_mask_loss(dynamicness):
    """Skewed binary entropy that pushes the dynamicness towards 0 or 1."""
    m = torch.clamp(dynamicness, min=1e-6, max=1.0 - 1e-6)
    return torch.mean(-((m**2) * torch.log(m**2) + (1 - m**2) * torch.log(1 - m**2)))


def masked_l1(pred, target, mask):
    return (
        torch.sum(torch.abs(pred - target) * mask)
        / (torch.sum(mask) + 1e-8)
        / pred.shape[-1]
    )


def distortion_loss(weights, z_vals):
    """Distortion loss from DVGO v2 / Mip-NeRF 360."""
    ray_id = torch.tile(
        torch.arange(weights.shape[0], dtype=torch.int64)[:, None],
        (1, weights.shape[1]),
    ).to(weights.device)
    return flatten_eff_distloss(
        torch.flatten(weights),
        torch.flatten(z_vals.detach()),
        1 / (weights.shape[1]),
        torch.flatten(ray_id),
    )


# ---------------------------------------------------------------------------
# Novel view rendering paths
# ---------------------------------------------------------------------------
def normalize(v):
    return v / np.linalg.norm(v)


def translate_pose(c2w, x_trans, y_trans, z_trans):
    i_pose = np.concatenate(
        [
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                axis=1,
            ),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
        ],
        axis=0,
    )
    i_pose = np.linalg.inv(i_pose)
    ref_pose = np.concatenate(
        [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
    )
    return np.dot(ref_pose, i_pose)[:3, :]


def generate_path(c2w, focal, sc, length):
    """Camera paths around a reference view (following DynamicNeRF).

    Returns {name: (poses, focals, change_time)}; change_time "change" means the
    time follows the path, otherwise the time is fixed.
    """
    max_disp = 48.0
    max_trans = max_disp / focal * sc
    paths = {}

    # fixed view, time changes
    paths["fix_view"] = [c2w[:3, :4] for _ in range(length)], [focal] * length

    # view and time change together
    poses = []
    for i in range(length):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans * 1.5 * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0) * 2.0 / 3.0
        )
        poses.append(translate_pose(c2w, x_trans, y_trans, 0.0))
    paths["change_view_time"] = poses, [focal] * length

    # dolly zoom
    poses, focals = [], []
    for i in range(30):
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        poses.append(translate_pose(c2w, 0.0, 0.0, z_trans))
        focals.append(focal - focal * 0.1 * z_trans / max_trans / 2.5)
    paths["dolly"] = poses, focals

    # zoom in
    poses = []
    for i in range(30):
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        poses.append(translate_pose(c2w, 0.0, 0.0, z_trans))
    paths["zoom"] = poses, [focal] * 30

    # spiral
    poses = []
    for i in range(30):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans * 1.5 * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0) * 2.0 / 3.0
        )
        poses.append(translate_pose(c2w, x_trans, y_trans, 0.0))
    paths["spiral"] = poses, [focal] * 30

    return {name: (np.stack(p, 0), f) for name, (p, f) in paths.items()}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_models(logfolder, expname, models, poses_mtx, focal_refine):
    for suffix, model in models.items():
        model.save(
            poses_mtx.detach().cpu(),
            focal_refine.detach().cpu(),
            f"{logfolder}/{expname}_{suffix}.th",
        )


def load_model(cls, path):
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt["kwargs"]
    poses_mtx = kwargs.pop("se3_poses").to(device)
    focal_refine = kwargs.pop("focal_ratio_refine").to(device)
    kwargs.update({"device": device})
    model = cls(**kwargs)
    model.load(ckpt)
    return model, poses_mtx, focal_refine


def model_size(name, model):
    num_params = sum(p.numel() for p in model.parameters())
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"[{name}] #Params: {num_params:,}, Size: {size / 1024 / 1024:.2f} MB")
    return num_params, size


@torch.no_grad()
def render_test(args):
    logfolder = os.path.join(args.basedir, args.expname)
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        use_foreground_mask=args.use_foreground_mask,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
    )
    set_num_frames(args, test_dataset)

    prefix = args.ckpt[:-3] if args.ckpt.endswith(".th") else args.ckpt
    tensorf_dynamic_g, _, _ = load_model(DynamicGeometryTensorVM, prefix + "_dynamic_g.th")
    tensorf_dynamic_c, _, _ = load_model(DynamicColorTensorVM, prefix + "_dynamic_c.th")
    tensorf_static, poses_mtx, focal_refine = load_model(StaticTensorVM, prefix + "_static.th")

    sizes = [
        model_size("dynamic_g", tensorf_dynamic_g),
        model_size("dynamic_c", tensorf_dynamic_c),
        model_size("static", tensorf_static),
    ]
    print(
        f"[Total] #Params: {sum(s[0] for s in sizes):,}, "
        f"Size: {sum(s[1] for s in sizes) / 1024 / 1024:.2f} MB"
    )

    os.makedirs(logfolder, exist_ok=True)
    np.save(f"{logfolder}/poses.npy", poses_mtx.detach().cpu().numpy())
    np.save(f"{logfolder}/focal.npy", focal_refine.detach().cpu().numpy())

    models = (tensorf_static, tensorf_dynamic_c, tensorf_dynamic_g)

    # reconstruction of the input video
    _, near_fars, depth_test_all = evaluation(
        test_dataset,
        poses_mtx,
        focal_refine.cpu(),
        *models,
        f"{logfolder}/imgs_test_all",
        N_samples=-1,
        ray_type=args.ray_type,
        device=device,
    )
    if not args.render_path:
        return

    # novel view / time synthesis
    render_idx = 0
    sc = near_fars[render_idx][0] * 0.75
    paths = generate_path(
        poses_mtx.cpu().detach().numpy()[render_idx],
        focal=focal_refine.item(),
        sc=sc,
        length=poses_mtx.shape[0],
    )
    fixed_time = render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0
    all_depths = {"imgs_test_all": depth_test_all}
    for name, (c2ws, focals) in paths.items():
        change_time = "change" if name in ("fix_view", "change_view_time") else fixed_time
        start_time = time.time()
        all_depths[name] = evaluation_path(
            test_dataset,
            *models,
            c2ws,
            focals,
            f"{logfolder}/{name}",
            N_samples=-1,
            ray_type=args.ray_type,
            change_time=change_time,
            device=device,
        )
        if name == "fix_view":
            elapsed = time.time() - start_time
            num_frames = len(all_depths[name])
            fps = num_frames / elapsed if elapsed > 0 else 0
            print(f"[Fix View] {num_frames} frames, {elapsed:.2f} sec, FPS: {fps:.2f}")

    # depth videos with a shared color range
    stacked = torch.stack([d for depths in all_depths.values() for d in depths])
    depth_map_min = torch.quantile(stacked[:, ::4, ::4], 0.05).item()
    depth_map_max = torch.quantile(stacked[:, ::4, ::4], 0.95).item()
    for name, depths in all_depths.items():
        frames = [
            visualize_depth_numpy(
                torch.clamp(d, min=depth_map_min, max=depth_map_max).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
            for d in depths
        ]
        imageio.mimwrite(
            f"{logfolder}/{name}/depthvideo.mp4",
            np.stack(frames),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, wireframe


def plot_cameras(ax, poses, color):
    vertices, wireframe = get_camera_mesh(poses, 0.005)
    center = vertices[:, -1]
    ax.scatter(center[:, 0], center[:, 1], center[:, 2], marker="o", color=color)
    for w in wireframe:
        ax.plot(w[:, 0].tolist(), w[:, 1].tolist(), w[:, 2].tolist(), color=color)
    return center


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    middle = limits.mean(1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([middle[0] - radius, middle[0] + radius])
    ax.set_ylim3d([middle[1] - radius, middle[1] + radius])
    ax.set_zlim3d([middle[2] - radius, middle[2] + radius])


def log_camera_poses(summary_writer, poses, gt_poses, iteration):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    if gt_poses is not None:
        center_gt = plot_cameras(ax, gt_poses, "C0")
    center = plot_cameras(ax, poses.cpu(), "C1")
    if gt_poses is not None:
        for i in range(center.shape[0]):
            ax.plot(
                [center_gt[i, 0], center[i, 0]],
                [center_gt[i, 1], center[i, 1]],
                [center_gt[i, 2], center[i, 2]],
                color="red",
            )
    set_axes_equal(ax)
    plt.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    img = (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32)
    summary_writer.add_image("camera_poses", img, iteration)
    plt.close(fig)


def log_test_views(summary_writer, vis, train_dataset, args, H, W, iteration):
    for tag, key, fmt in [
        ("rgb_maps", "rgb_maps", "NHWC"),
        ("rgb_maps_s", "rgb_maps_s", "NHWC"),
        ("rgb_maps_d", "rgb_maps_d", "NHWC"),
        ("depth_map", "depth_maps", "NCHW"),
        ("depth_map_s", "depth_maps_s", "NCHW"),
        ("depth_map_d", "depth_maps_d", "NCHW"),
        ("monodepth_tb", "monodepth", "NCHW"),
        ("blending_maps", "blending_maps", "NCHW"),
        ("gt_maps", "gt_rgbs", "NHWC"),
        ("induced_flow_f", "induced_flow_f", "NHWC"),
        ("induced_flow_b", "induced_flow_b", "NHWC"),
        ("induced_flow_s_f", "induced_flow_s_f", "NHWC"),
        ("induced_flow_s_b", "induced_flow_s_b", "NHWC"),
        ("delta_xyz_tb", "delta_xyz", "NHWC"),
    ]:
        summary_writer.add_images(
            "test/" + tag, torch.stack(vis[key], 0), global_step=iteration, dataformats=fmt
        )

    for tag, flows in [
        ("gt_flow_f", train_dataset.all_flows_f),
        ("gt_flow_b", train_dataset.all_flows_b),
    ]:
        flows = flows.view(args.N_voxel_t, H, W, 2)
        summary_writer.add_images(
            "test/" + tag,
            torch.stack([torch.from_numpy(flow_to_image(f.numpy())) for f in flows], 0),
            global_step=iteration,
            dataformats="NHWC",
        )
    summary_writer.add_images(
        "test/gt_blending_maps",
        train_dataset.all_foreground_masks.view(args.N_voxel_t, H, W, 3),
        global_step=iteration,
        dataformats="NHWC",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def reconstruction(args):
    check_hyperparameters(args)

    dataset = dataset_dict[args.dataset_name]
    dataset_kwargs = dict(
        downsample=args.downsample_train,
        use_foreground_mask=args.use_foreground_mask,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
    )
    train_dataset = dataset(args.datadir, split="train", is_stack=False, **dataset_kwargs)
    test_dataset = dataset(args.datadir, split="test", is_stack=True, **dataset_kwargs)
    set_num_frames(args, train_dataset)
    near_far = train_dataset.near_far
    W, H = train_dataset.img_wh
    ray_type = args.ray_type

    upsamp_list = args.upsamp_list

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # ---------------- models ----------------
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    model_kwargs = dict(
        density_n_comp=args.n_lamb_sigma,
        appearance_n_comp=args.n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        view_pe=args.view_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )
    # static field (+ rigidity)
    tensorf_static = StaticTensorVM(
        aabb,
        reso_cur,
        args.N_voxel_t,
        device,
        shadingMode=args.shadingModeStatic,
        fea_pe=args.fea_pe_static,
        **model_kwargs,
    )
    # dynamic color field
    tensorf_dynamic_c = DynamicColorTensorVM(
        aabb,
        reso_cur,
        args.N_voxel_t,
        device,
        shadingMode=args.shadingMode,
        fea_pe=args.fea_pe_dynamic,
        **model_kwargs,
    )
    # dynamic geometry field (+ scene flow)
    tensorf_dynamic_g = DynamicGeometryTensorVM(
        aabb,
        reso_cur,
        args.N_voxel_t,
        device,
        shadingMode=args.shadingMode,
        fea_pe=args.fea_pe_dynamic,
        **model_kwargs,
    )
    models = {
        "dynamic_g": tensorf_dynamic_g,
        "dynamic_c": tensorf_dynamic_c,
        "static": tensorf_static,
    }

    grad_vars = tensorf_static.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars.extend(tensorf_dynamic_c.get_optparam_groups(args.lr_init, args.lr_basis))
    grad_vars.extend(tensorf_dynamic_g.get_optparam_groups(args.lr_init, args.lr_basis))
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # coarse-to-fine: grid resolutions, linear in log space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]

    # ---------------- data ----------------
    allrgbs = train_dataset.all_rgbs
    allts = train_dataset.all_ts
    allposes = train_dataset.all_poses if args.with_GT_poses else None  # (N, 3, 4)
    allflows_f = train_dataset.all_flows_f.to(device)
    allflowmasks_f = train_dataset.all_flow_masks_f.to(device)
    allflows_b = train_dataset.all_flows_b.to(device)
    allflowmasks_b = train_dataset.all_flow_masks_b.to(device)
    alldisps = train_dataset.all_disps
    allforegroundmasks = train_dataset.all_foreground_masks

    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).to(device)
    allgrids = torch.tile(torch.unsqueeze(grid, 0), (args.N_voxel_t, 1, 1, 1)).view(-1, 2)

    # ---------------- cameras ----------------
    # 6D rotation + translation per frame
    init_poses = torch.zeros(args.N_voxel_t, 9)
    if args.with_GT_poses:
        init_poses[..., 0:3] = allposes[..., :, 0]
        init_poses[..., 3:6] = allposes[..., :, 1]
        init_poses[..., 6:9] = allposes[..., :, 3]
    else:
        init_poses[..., 0] = 1
        init_poses[..., 4] = 1
    poses_refine = torch.nn.Embedding(args.N_voxel_t, 9).to(device)
    poses_refine.weight.data.copy_(init_poses.to(device))

    # focal length parameterized by the field of view
    fov_refine_embedding = torch.nn.Embedding(1, 1).to(device)
    fov_refine_embedding.weight.data.copy_(
        torch.ones(1, 1).to(device) * args.init_fov_deg / 180 * np.pi
    )
    if not args.optimize_focal_length:
        focal_refine = torch.tensor(train_dataset.focal[0]).to(device)

    pose_lr_gamma = (args.lr_pose_final / args.lr_pose) ** (
        1.0 / (args.pose_freeze_iter - upsamp_list[-1])
    )
    if args.optimize_poses:
        optimizer_pose = torch.optim.Adam(poses_refine.parameters(), lr=args.lr_pose)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pose, gamma=pose_lr_gamma)
    if args.optimize_focal_length:
        optimizer_focal = torch.optim.Adam(fov_refine_embedding.parameters(), lr=0.0)
        scheduler_focal = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_focal, gamma=pose_lr_gamma
        )

    trainingSampler = SimpleSampler(allts.shape[0], args.batch_size)
    trainingSampler_2 = SimpleSampler(allts.shape[0], args.batch_size)

    L1_reg_weight = args.L1_weight_inital
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    distortion_weight_static = args.distortion_weight_static
    distortion_weight_dynamic = args.distortion_weight_dynamic
    tvreg = TVLoss()

    PSNRs = []
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        # loss schedules
        decay_static = 1.0 / (10 ** (iteration / args.static_decay_iters))
        decay_dynamic = 1.0 / (10 ** (iteration // args.dynamic_decay_iters))
        decay_mask_smooth = 1.0 / (10 ** (iteration // args.mask_smooth_decay_iters))

        if args.optimize_focal_length:
            focal_refine = (
                np.maximum(H, W) / 2.0 / torch.tan(fov_refine_embedding.weight[0, 0])
            )

        # ---------------- batch ----------------
        ray_idx = trainingSampler.nextids()
        rgb_train, ts_train, grid_train = (
            allrgbs[ray_idx].to(device),
            allts[ray_idx].to(device),
            allgrids[ray_idx],
        )
        flow_f_train, flow_mask_f_train, flow_b_train, flow_mask_b_train = (
            allflows_f[ray_idx],
            allflowmasks_f[ray_idx][..., None],
            allflows_b[ray_idx],
            allflowmasks_b[ray_idx][..., None],
        )
        disps_train = alldisps[ray_idx].to(device)
        foregroundmasks_train = allforegroundmasks[ray_idx].to(device)
        static_mask_train = 1.0 - foregroundmasks_train[..., 0:1]

        poses_mtx = pose_to_mtx(poses_refine.weight)

        i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))
        directions = get_ray_directions_lean(
            i, j, [focal_refine, focal_refine], [W / 2, H / 2]
        )
        poses_mtx_batched = poses_mtx[view_ids]
        rays_train = build_rays(directions, poses_mtx_batched, H, W, focal_refine, ray_type)

        t_ref = ray_idx // (H * W)
        u_ref = (ray_idx % (H * W)) // W  # height
        v_ref = (ray_idx % (H * W)) % W  # width
        t_interval = 2 / (args.N_voxel_t - 1)

        # poses of the next / previous frame
        poses_f_train = torch.cat((poses_mtx[1:], poses_mtx[-1:]), 0)[t_ref]
        poses_b_train = torch.cat((poses_mtx[0:1], poses_mtx[:-1]), 0)[t_ref]

        total_loss = 0.0

        # ======== composite rendering at the training time ========
        # (camera gradients are blocked; the static field is trained below)
        out = render_composite(
            tensorf_static, tensorf_dynamic_g, tensorf_dynamic_c,
            rays_train.detach(), ts_train, nSamples, ray_type,
            is_train=True, detach_static=True,
        )

        # ======== novel-time regularization ========
        # the same rays rendered at a random time
        ray_idx_rand = trainingSampler_2.nextids()
        ts_train_rand = allts[ray_idx_rand].to(device)
        out_rand = render_composite(
            tensorf_static, tensorf_dynamic_g, tensorf_dynamic_c,
            rays_train.detach(), ts_train_rand, nSamples, ray_type,
            is_train=True, detach_static=True,
        )

        if iteration >= args.mask_reg_start_iter:
            skewed_mask_loss_rand = skewed_mask_loss(out_rand.dynamicness_map)
            total_loss += args.w_skewed_mask * skewed_mask_loss_rand
            summary_writer.add_scalar("train/skewed_mask_loss_rand", skewed_mask_loss_rand.detach().item(), global_step=iteration)

            novel_view_time_mask_loss = torch.mean(torch.abs(out_rand.dynamicness_map))
            total_loss += args.w_mask_l1 * novel_view_time_mask_loss
            summary_writer.add_scalar("train/novel_view_time_mask_loss", novel_view_time_mask_loss.detach().item(), global_step=iteration)

        novel_order_loss = order_loss(out_rand.depth_map_d, out_rand.depth_map_s, out_rand.dynamicness_map, ray_type)
        total_loss += novel_order_loss * args.w_order
        summary_writer.add_scalar("train/novel_order_loss", novel_order_loss.detach().item(), global_step=iteration)

        if distortion_weight_dynamic > 0:
            loss_distortion = distortion_loss(out_rand.weights_d, out_rand.z_vals)
            total_loss += loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            summary_writer.add_scalar("train/loss_distortion_rand", loss_distortion.detach().item(), global_step=iteration)

        # ======== photometric losses ========
        scene_flow_f, scene_flow_b = tensorf_dynamic_g.get_forward_backward_scene_flow(out.pts, ts_train)

        loss = torch.mean((out.rgb_map - rgb_train) ** 2)
        PSNRs.append(-10.0 * np.log(loss.detach().item()) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar("train/mse", loss.detach().item(), global_step=iteration)
        total_loss += args.w_rgb * loss

        img_d_loss = torch.mean((out.rgb_map_d - rgb_train) ** 2)
        total_loss += args.w_rgb_dynamic * img_d_loss
        summary_writer.add_scalar("train/img_d_loss", img_d_loss.detach().item(), global_step=iteration)

        # ======== dynamicness (rigidity) supervision ========
        if iteration >= args.mask_loss_start_iter:
            mask_loss = torch.mean(torch.abs(out.dynamicness_map - foregroundmasks_train[..., 0]))
            total_loss += args.w_mask * mask_loss * decay_mask_smooth
            summary_writer.add_scalar("train/mask_loss", mask_loss.detach().item(), global_step=iteration)

        if iteration >= args.mask_reg_start_iter:
            skewed_loss = skewed_mask_loss(out.dynamicness_map)
            total_loss += args.w_skewed_mask * skewed_loss
            summary_writer.add_scalar("train/skewed_mask_loss", skewed_loss.detach().item(), global_step=iteration)

            mask_L1_reg_loss = torch.mean(torch.abs(out.dynamicness_map))
            total_loss += args.w_mask_l1 * mask_L1_reg_loss
            summary_writer.add_scalar("train/mask_L1_reg_loss", mask_L1_reg_loss.detach().item(), global_step=iteration)

        # ======== scene flow: induced optical flow ========
        if ray_type == "ndc":
            pts_f = out.pts + scene_flow_f
            pts_b = out.pts + scene_flow_b
        elif ray_type == "contract":
            pts_f = torch.clamp(out.pts + scene_flow_f, min=-2.0 + 1e-6, max=2.0 - 1e-6)
            pts_b = torch.clamp(out.pts + scene_flow_b, min=-2.0 + 1e-6, max=2.0 - 1e-6)

        induced_flow_f, induced_disp_f = induce_flow(
            H, W, focal_refine.detach(), poses_f_train.detach(), out.weights_d,
            pts_f, grid_train, rays_train.detach(), ray_type=ray_type,
        )
        flow_f_loss = masked_l1(induced_flow_f, flow_f_train, flow_mask_f_train)
        total_loss += args.w_flow * flow_f_loss * decay_dynamic
        induced_flow_b, induced_disp_b = induce_flow(
            H, W, focal_refine.detach(), poses_b_train.detach(), out.weights_d,
            pts_b, grid_train, rays_train.detach(), ray_type=ray_type,
        )
        flow_b_loss = masked_l1(induced_flow_b, flow_b_train, flow_mask_b_train)
        total_loss += args.w_flow * flow_b_loss * decay_dynamic
        summary_writer.add_scalar("train/flow_f_loss", flow_f_loss.detach().item(), global_step=iteration)
        summary_writer.add_scalar("train/flow_b_loss", flow_b_loss.detach().item(), global_step=iteration)

        small_scene_flow_loss = torch.mean(torch.abs(scene_flow_f)) + torch.mean(torch.abs(scene_flow_b))
        total_loss += args.small_scene_flow_weight * small_scene_flow_loss
        summary_writer.add_scalar("train/small_scene_flow_loss", small_scene_flow_loss.detach().item(), global_step=iteration)

        # ======== scene flow: disparity consistency with the neighbor frames ========
        neighbor_outs = []
        for flow_train, flow_mask_train, poses_n_train, dt, induced_disp, name in [
            (flow_f_train, flow_mask_f_train, poses_f_train, t_interval, induced_disp_f, "disp_f_loss"),
            (flow_b_train, flow_mask_b_train, poses_b_train, -t_interval, induced_disp_b, "disp_b_loss"),
        ]:
            uv = torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_train.device) + flow_train
            rays_n = build_rays(
                uv_to_directions(uv, focal_refine.detach(), W, H),
                poses_n_train, H, W, focal_refine.detach(), ray_type,
            )
            out_n = render_composite(
                tensorf_static, tensorf_dynamic_g, tensorf_dynamic_c,
                rays_n.detach(), ts_train + dt, nSamples, ray_type,
                is_train=True, detach_static=True,
            )
            _, induced_disp_n = induce_flow(
                H, W, focal_refine.detach(), poses_n_train.detach(), out_n.weights_d,
                out_n.pts, grid_train, rays_n.detach(), ray_type=ray_type,
            )
            disp_loss = masked_l1(induced_disp, induced_disp_n, flow_mask_train)
            total_loss += args.w_disp * disp_loss * decay_dynamic
            summary_writer.add_scalar("train/" + name, disp_loss.detach().item(), global_step=iteration)
            neighbor_outs.append(out_n)
        out_f, out_b = neighbor_outs

        smooth_scene_flow_loss = torch.mean(torch.abs(scene_flow_f + scene_flow_b))
        total_loss += smooth_scene_flow_loss * args.smooth_scene_flow_weight
        summary_writer.add_scalar("train/smooth_scene_flow_loss", smooth_scene_flow_loss.detach().item(), global_step=iteration)

        # ======== monocular depth (dynamic) ========
        total_mono_depth_loss = monodepth_loss(out.depth_map_d, disps_train, t_ref, args.N_voxel_t, ray_type)
        total_loss += total_mono_depth_loss * args.monodepth_weight_dynamic * decay_dynamic
        summary_writer.add_scalar("train/total_mono_depth_loss_dynamic", total_mono_depth_loss.detach().item(), global_step=iteration)

        # ======== depth order between static and dynamic ========
        order_loss_train = order_loss(out.depth_map_d, out.depth_map_s, out.dynamicness_map, ray_type)
        total_loss += order_loss_train * args.w_order
        summary_writer.add_scalar("train/order_loss", order_loss_train.detach().item(), global_step=iteration)

        if distortion_weight_dynamic > 0:
            loss_distortion = distortion_loss(out.weights_d, out.z_vals)
            loss_distortion += distortion_loss(out_f.weights_d, out_f.z_vals)
            loss_distortion += distortion_loss(out_b.weights_d, out_b.z_vals)
            total_loss += loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            summary_writer.add_scalar("train/loss_distortion", loss_distortion.detach().item(), global_step=iteration)

        # ======== grid regularizers (dynamic) ========
        if L1_reg_weight > 0:
            loss_reg_L1_density = tensorf_dynamic_g.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1_density
            summary_writer.add_scalar("train/loss_reg_L1_density", loss_reg_L1_density.detach().item(), global_step=iteration)
        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf_dynamic_g.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar("train/reg_tv_density", loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf_dynamic_c.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar("train/reg_tv_app", loss_tv.detach().item(), global_step=iteration)

        # ======== static field (receives camera gradients) ========
        out_s = render_composite(
            tensorf_static, tensorf_dynamic_g, tensorf_dynamic_c,
            rays_train, ts_train, nSamples, ray_type, is_train=True,
        )

        img_s_loss = (
            torch.sum((out_s.rgb_map_s - rgb_train) ** 2 * static_mask_train)
            / (torch.sum(static_mask_train) + 1e-8)
            / out_s.rgb_map_s.shape[-1]
        )
        total_loss += args.w_rgb_static * img_s_loss
        summary_writer.add_scalar("train/img_s_loss", img_s_loss.detach().item(), global_step=iteration)

        if distortion_weight_static > 0:
            loss_distortion_static = distortion_loss(out_s.weights_s, out_s.z_vals)
            total_loss += loss_distortion_static * distortion_weight_static * (iteration / args.n_iters)
            summary_writer.add_scalar("train/loss_distortion_static", loss_distortion_static.detach().item(), global_step=iteration)

        # ======== grid regularizers (static) ========
        if L1_reg_weight > 0:
            loss_reg_L1_density_s = tensorf_static.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1_density_s
            summary_writer.add_scalar("train/loss_reg_L1_density_s", loss_reg_L1_density_s.detach().item(), global_step=iteration)
            loss_tv_static = tensorf_static.TV_loss_blending(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar("train/reg_tv_blending", loss_tv_static.detach().item(), global_step=iteration)
        if TV_weight_density > 0:
            loss_tv_static = tensorf_static.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar("train/reg_tv_density_static", loss_tv_static.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            loss_tv_static = tensorf_static.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar("train/reg_tv_app_static", loss_tv_static.detach().item(), global_step=iteration)

        summary_writer.add_scalar("train/focal_ratio_refine", focal_refine.detach().item(), global_step=iteration)

        # ======== camera pose estimation (static region) ========
        if args.optimize_poses:
            static_flow_mask_f = flow_mask_f_train * static_mask_train
            static_flow_mask_b = flow_mask_b_train * static_mask_train

            # induced flow of the static field = camera motion
            induced_flow_f_s, induced_disp_f_s = induce_flow(
                H, W, focal_refine, poses_f_train, out_s.weights_s, out_s.pts,
                grid_train, rays_train, ray_type=ray_type,
            )
            flow_f_s_loss = masked_l1(induced_flow_f_s, flow_f_train, static_flow_mask_f)
            total_loss += args.w_flow_static * flow_f_s_loss * decay_static
            induced_flow_b_s, induced_disp_b_s = induce_flow(
                H, W, focal_refine, poses_b_train, out_s.weights_s, out_s.pts,
                grid_train, rays_train, ray_type=ray_type,
            )
            flow_b_s_loss = masked_l1(induced_flow_b_s, flow_b_train, static_flow_mask_b)
            total_loss += args.w_flow_static * flow_b_s_loss * decay_static
            summary_writer.add_scalar("train/flow_f_s_loss", flow_f_s_loss.detach().item(), global_step=iteration)
            summary_writer.add_scalar("train/flow_b_s_loss", flow_b_s_loss.detach().item(), global_step=iteration)

            # static disparity consistency with the neighbor frames
            for flow_train, static_flow_mask, poses_n_train, induced_disp_s, name in [
                (flow_f_train, static_flow_mask_f, poses_f_train, induced_disp_f_s, "disp_f_s_loss"),
                (flow_b_train, static_flow_mask_b, poses_b_train, induced_disp_b_s, "disp_b_s_loss"),
            ]:
                uv = torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_train.device) + flow_train
                rays_n = build_rays(
                    uv_to_directions(uv, focal_refine, W, H), poses_n_train, H, W, focal_refine, ray_type
                )
                xyz_sampled_n, z_vals_n, ray_valid_n = sampleXYZ(
                    tensorf_static, rays_n, N_samples=nSamples, ray_type=ray_type, is_train=True
                )
                _, weights_s_n, _, _ = tensorf_static(rays_n, ts_train, xyz_sampled_n, z_vals_n, ray_valid_n)
                _, induced_disp_s_n = induce_flow(
                    H, W, focal_refine, poses_n_train, weights_s_n, xyz_sampled_n,
                    grid_train, rays_n, ray_type=ray_type,
                )
                disp_s_loss = masked_l1(induced_disp_s, induced_disp_s_n, static_flow_mask)
                total_loss += args.w_disp_static * disp_s_loss * decay_static
                summary_writer.add_scalar("train/" + name, disp_s_loss.detach().item(), global_step=iteration)

            # monocular depth in the static region
            total_mono_depth_loss = monodepth_loss(
                out_s.depth_map_s, disps_train, t_ref, args.N_voxel_t, ray_type,
                valid_mask=foregroundmasks_train[..., 0].cpu() < 0.5,
            )
            total_loss += total_mono_depth_loss * args.monodepth_weight_static * decay_static
            summary_writer.add_scalar("train/total_mono_depth_loss_static", total_mono_depth_loss.detach().item(), global_step=iteration)

            # static disparity smoothness between neighboring pixels
            i_neighbor = torch.clamp(i + 1, max=W - 1)
            j_neighbor = torch.clamp(j + 1, max=H - 1)
            rays_i_neighbor = build_rays(
                get_ray_directions_lean(i_neighbor, j, [focal_refine, focal_refine], [W / 2, H / 2]),
                poses_mtx_batched, H, W, focal_refine, ray_type,
            )
            rays_j_neighbor = build_rays(
                get_ray_directions_lean(i, j_neighbor, [focal_refine, focal_refine], [W / 2, H / 2]),
                poses_mtx_batched, H, W, focal_refine, ray_type,
            )
            disp_s = 1.0 / torch.clamp(out_s.depth_map_s, min=1e-6)
            disp_smooth_loss = 0.0
            for rays_neighbor in [rays_i_neighbor, rays_j_neighbor]:
                out_neighbor = render_composite(
                    tensorf_static, tensorf_dynamic_g, tensorf_dynamic_c,
                    rays_neighbor, ts_train, nSamples, ray_type, is_train=True,
                )
                disp_s_neighbor = 1.0 / torch.clamp(out_neighbor.depth_map_s, min=1e-6)
                disp_smooth_loss = disp_smooth_loss + torch.mean((disp_s - disp_s_neighbor) ** 2)
            total_loss += disp_smooth_loss * args.w_disp_smooth * decay_mask_smooth
            summary_writer.add_scalar("train/disp_smooth_loss", disp_smooth_loss.detach().item(), global_step=iteration)

        # ======== optimization step ========
        if args.optimize_poses:
            optimizer_pose.zero_grad()
        if args.optimize_focal_length:
            optimizer_focal.zero_grad()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if args.optimize_poses:
            optimizer_pose.step()
            scheduler.step()
        if args.optimize_focal_length:
            optimizer_focal.step()
            scheduler_focal.step()

        summary_writer.add_scalar("train/lr_grid", optimizer.param_groups[0]["lr"], global_step=iteration)
        if args.optimize_poses:
            summary_writer.add_scalar("train/lr_pose", optimizer_pose.param_groups[0]["lr"], global_step=iteration)
        if args.optimize_focal_length:
            summary_writer.add_scalar("train/lr_focal", optimizer_focal.param_groups[0]["lr"], global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(f"Iteration {iteration:05d}: train_psnr = {float(np.mean(PSNRs)):.2f}")
            PSNRs = []
            log_camera_poses(summary_writer, poses_mtx.detach(), allposes, iteration)
            save_models(logfolder, args.expname, models, poses_mtx, focal_refine)

        if iteration % args.vis_train_every == args.vis_train_every - 1 and args.N_vis != 0:
            vis = render(
                test_dataset,
                poses_mtx,
                focal_refine.cpu(),
                tensorf_static,
                tensorf_dynamic_c,
                tensorf_dynamic_g,
                N_vis=args.N_vis,
                N_samples=nSamples,
                ray_type=ray_type,
                device=device,
            )
            log_test_views(summary_writer, vis, train_dataset, args, H, W, iteration)

        # ======== coarse-to-fine grid upsampling ========
        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf_dynamic_g.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf_dynamic_g.upsample_volume_grid(reso_cur)
            tensorf_dynamic_c.upsample_volume_grid(reso_cur)
            tensorf_static.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                lr_scale = 1
                if args.optimize_poses:
                    optimizer_pose.param_groups[0]["lr"] = args.lr_pose
                if iteration >= args.focal_opt_start_iter and args.optimize_focal_length:
                    optimizer_focal.param_groups[0]["lr"] = args.lr_pose
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf_static.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            grad_vars.extend(
                tensorf_dynamic_g.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            )
            grad_vars.extend(
                tensorf_dynamic_c.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # camera parameters are fixed in the later stage of training
        if iteration > args.pose_freeze_iter:
            if args.optimize_poses:
                optimizer_pose.param_groups[0]["lr"] = 0.0
            if args.optimize_focal_length:
                optimizer_focal.param_groups[0]["lr"] = 0.0

    save_models(logfolder, args.expname, models, poses_mtx, focal_refine)

    PSNRs_all, near_fars, _ = evaluation(
        test_dataset,
        poses_mtx,
        focal_refine.cpu(),
        tensorf_static,
        tensorf_dynamic_c,
        tensorf_dynamic_g,
        f"{logfolder}/imgs_test_all",
        N_samples=-1,
        ray_type=ray_type,
        device=device,
    )
    print(f"======> {args.expname} psnr (all frames): {np.mean(PSNRs_all)} <========================")

    # save the estimated cameras in LLFF format
    poses_saving = poses_mtx.clone()
    poses_saving = torch.cat(
        [-poses_saving[..., 1:2], poses_saving[..., :1], poses_saving[..., 2:4]], -1
    )
    hwf = (
        torch.from_numpy(np.array([H, W, focal_refine.detach().cpu()]))
        * args.downsample_train
    )
    hwf = torch.stack([hwf] * args.N_voxel_t, 0)[..., None]
    poses_saving = torch.cat([poses_saving.cpu(), hwf], -1).view(args.N_voxel_t, -1)
    poses_bounds_saving = (
        torch.cat([poses_saving, torch.from_numpy(np.array(near_fars))], -1)
        .detach()
        .numpy()
    )
    np.save(os.path.join(logfolder, "poses_bounds_estimated.npy"), poses_bounds_saving)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
