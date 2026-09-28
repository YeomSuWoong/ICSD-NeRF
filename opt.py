# Copyright (c) Meta Platforms, Inc. and affiliates.

import configargparse

# Hyperparameter values are not included in this release. The options below
# have no default and have to be set in the config file before training.
HYPERPARAMETERS = [
    # optimization
    "n_iters",
    "batch_size",
    "lr_init",
    "lr_basis",
    "lr_decay_target_ratio",
    "lr_pose",
    "lr_pose_final",
    "init_fov_deg",
    "pose_freeze_iter",
    "focal_opt_start_iter",
    # grid / sampling
    "N_voxel_init",
    "N_voxel_final",
    "upsamp_list",
    "step_ratio",
    # network capacity
    "n_lamb_sigma",
    "n_lamb_sh",
    "data_dim_color",
    "featureC",
    "view_pe",
    "fea_pe_static",
    "fea_pe_dynamic",
    # loss weights
    "w_rgb",
    "w_rgb_static",
    "w_rgb_dynamic",
    "w_mask",
    "w_skewed_mask",
    "w_mask_l1",
    "w_order",
    "w_flow",
    "w_disp",
    "w_flow_static",
    "w_disp_static",
    "w_disp_smooth",
    "monodepth_weight_static",
    "monodepth_weight_dynamic",
    "smooth_scene_flow_weight",
    "small_scene_flow_weight",
    "distortion_weight_static",
    "distortion_weight_dynamic",
    "TV_weight_density",
    "TV_weight_app",
    "L1_weight_inital",
    # loss schedules
    "mask_loss_start_iter",
    "mask_reg_start_iter",
    "static_decay_iters",
    "dynamic_decay_iters",
    "mask_smooth_decay_iters",
]


def check_hyperparameters(args):
    missing = [name for name in HYPERPARAMETERS if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "The following hyperparameters are not set (their values are not "
            "included in this release; set them in the config file): "
            + ", ".join(missing)
        )


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./log", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--add_timestamp", type=int, default=0, help="add timestamp to dir"
    )
    parser.add_argument("--datadir", type=str, help="input data directory")
    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=10,
        help="how many iterations to show psnrs or iters",
    )

    # dataset
    parser.add_argument(
        "--dataset_name", type=str, default="nvidia", choices=["nvidia", "davis"]
    )
    parser.add_argument("--downsample_train", type=float, default=1.0)
    parser.add_argument(
        "--N_voxel_t",
        type=int,
        default=None,
        help="number of frames (inferred from the dataset if not given)",
    )
    parser.add_argument(
        "--use_foreground_mask",
        type=str,
        default="motion_masks",
        choices=["motion_masks", "epipolar_motion_masks", "epipolar_error_png"],
        help="folder of the motion masks used to supervise the dynamicness map",
    )
    parser.add_argument(
        "--with_GT_poses", type=int, default=0, help="initialize from poses_bounds.npy"
    )

    # camera optimization
    parser.add_argument("--optimize_poses", type=int, default=0)
    parser.add_argument("--optimize_focal_length", type=int, default=0)
    parser.add_argument("--lr_pose", type=float, default=None)
    parser.add_argument("--lr_pose_final", type=float, default=None)
    parser.add_argument(
        "--init_fov_deg", type=float, default=None, help="initial field of view"
    )
    parser.add_argument(
        "--pose_freeze_iter",
        type=int,
        default=None,
        help="camera poses / focal length are fixed after this iteration",
    )
    parser.add_argument(
        "--focal_opt_start_iter",
        type=int,
        default=None,
        help="focal length optimization starts at the first grid upsampling after this iteration",
    )

    # training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--lr_init", type=float, default=None, help="lr of the grids")
    parser.add_argument("--lr_basis", type=float, default=None, help="lr of the MLPs")
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=-1,
        help="number of iterations the lr will decay to the target ratio; -1 will set it to n_iters",
    )
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=None,
        help="the target decay ratio; after decay_iters inital lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to inital after upsampling",
    )

    # loss weights
    parser.add_argument("--w_rgb", type=float, default=None, help="composited RGB")
    parser.add_argument("--w_rgb_static", type=float, default=None, help="static RGB (static region)")
    parser.add_argument("--w_rgb_dynamic", type=float, default=None, help="dynamic RGB")
    parser.add_argument("--w_mask", type=float, default=None, help="dynamicness vs. motion mask")
    parser.add_argument("--w_skewed_mask", type=float, default=None, help="skewed binary entropy on dynamicness")
    parser.add_argument("--w_mask_l1", type=float, default=None, help="L1 sparsity on dynamicness")
    parser.add_argument("--w_order", type=float, default=None, help="static/dynamic depth order")
    parser.add_argument("--w_flow", type=float, default=None, help="induced flow (dynamic)")
    parser.add_argument("--w_disp", type=float, default=None, help="disparity consistency (dynamic)")
    parser.add_argument("--w_flow_static", type=float, default=None, help="induced flow (static, pose)")
    parser.add_argument("--w_disp_static", type=float, default=None, help="disparity consistency (static, pose)")
    parser.add_argument("--w_disp_smooth", type=float, default=None, help="static disparity smoothness")
    parser.add_argument("--monodepth_weight_static", type=float, default=None)
    parser.add_argument("--monodepth_weight_dynamic", type=float, default=None)
    parser.add_argument("--smooth_scene_flow_weight", type=float, default=None)
    parser.add_argument("--small_scene_flow_weight", type=float, default=None)
    parser.add_argument("--distortion_weight_static", type=float, default=None)
    parser.add_argument("--distortion_weight_dynamic", type=float, default=None)
    parser.add_argument("--TV_weight_density", type=float, default=None)
    parser.add_argument("--TV_weight_app", type=float, default=None)
    parser.add_argument("--L1_weight_inital", type=float, default=None)

    # loss schedules
    parser.add_argument(
        "--mask_loss_start_iter", type=int, default=None,
        help="iteration from which the motion-mask loss is used",
    )
    parser.add_argument(
        "--mask_reg_start_iter", type=int, default=None,
        help="iteration from which the skewed-entropy / L1 dynamicness losses are used",
    )
    parser.add_argument(
        "--static_decay_iters", type=float, default=None,
        help="static (pose) losses decay by 10x every this many iterations (continuous)",
    )
    parser.add_argument(
        "--dynamic_decay_iters", type=int, default=None,
        help="flow / disparity / mono-depth losses of the dynamic field decay by 10x every this many iterations (step)",
    )
    parser.add_argument(
        "--mask_smooth_decay_iters", type=int, default=None,
        help="motion-mask and disparity-smoothness losses decay by 10x every this many iterations (step)",
    )

    # model
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=None)
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-10,
        help="shift density in softplus; making density = 0  when feature == 0",
    )
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument("--N_voxel_init", type=int, default=None)
    parser.add_argument("--N_voxel_final", type=int, default=None)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--step_ratio", type=float, default=None)
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1000000,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument(
        "--ray_type",
        type=str,
        default="ndc",
        choices=["ndc", "contract"],
        help="ndc: forward-facing scenes, contract: unbounded scenes",
    )

    # network decoder
    parser.add_argument(
        "--shadingMode",
        type=str,
        default="MLP_Fea_late_view",
        help="color decoder of the dynamic field",
    )
    parser.add_argument(
        "--shadingModeStatic",
        type=str,
        default="MLP_Fea",
        help="color decoder of the static field",
    )
    parser.add_argument("--view_pe", type=int, default=None, help="number of pe for view")
    parser.add_argument("--fea_pe_static", type=int, default=None, help="number of pe for features (static)")
    parser.add_argument("--fea_pe_dynamic", type=int, default=None, help="number of pe for features (dynamic)")
    parser.add_argument(
        "--featureC", type=int, default=None, help="hidden feature channel in MLP"
    )

    # rendering
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint prefix, e.g. log/nvidia/Balloon1/Balloon1(.th)",
    )
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)

    # logging
    parser.add_argument("--N_vis", type=int, default=5, help="N images to vis")
    parser.add_argument(
        "--vis_train_every",
        type=int,
        default=2000,
        help="frequency of visualize the training on tensorboard",
    )

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
