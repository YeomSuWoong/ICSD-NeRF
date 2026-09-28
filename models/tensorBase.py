# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


class MLPRender_Fea(torch.nn.Module):
    """Color decoder of the static field: (feature, view direction) -> RGB."""

    def __init__(self, inChanel, viewpe, feape, featureC):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, time):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_Fea_late_view(torch.nn.Module):
    """Color decoder of the dynamic field.

    Feature, (canonical) position and time are encoded first; the view direction
    is injected only in the last layer.
    """

    def __init__(self, inChanel, viewpe, feape, featureC):
        super(MLPRender_Fea_late_view, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel + 2 * 10 * 3 + 3 + 2 * 8 * 1 + 1
        self.in_view = 2 * viewpe * 3 + 3
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(self, pts, viewdirs, features, time):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        indata += [pts]
        indata += [positional_encoding(pts, 10)]
        indata += [time]
        indata += [positional_encoding(time, 8)]
        indata_view = [viewdirs.detach()]
        if self.viewpe > 0:
            indata_view += [positional_encoding(viewdirs.detach(), self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view, dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    """Shared logic of the VM-decomposed (TensoRF) fields."""

    def __init__(
        self,
        aabb,
        gridSize,
        tSize,
        device,
        density_n_comp,
        appearance_n_comp,
        app_dim,
        shadingMode,
        near_far,
        density_shift,
        distance_scale,
        view_pe,
        fea_pe,
        featureC,
        step_ratio,
        fea2denseAct,
        rayMarch_weight_thres=0.0001,
        # kept only so that checkpoints saved by earlier versions can be loaded
        alphaMask_thres=None,
        pos_pe=None,
    ):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio
        self.tSize = int(tSize)

        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = (
            shadingMode,
            pos_pe,
            view_pe,
            fea_pe,
            featureC,
        )
        self.init_render_func(shadingMode, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_late_view":
            self.renderModule = MLPRender_Fea_late_view(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        else:
            raise ValueError(f"Unknown shadingMode: {shadingMode}")

    def update_stepSize(self, gridSize):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print(f"grid size: {gridSize}, sampling number: {self.nSamples}")

    def init_svd_volume(self, res, device):
        raise NotImplementedError

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                )
            )
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def vm_features(self, planes, lines, xyz_sampled, strides=(1,)):
        """Sample plane x line (VM) features at normalized points.

        With several strides, the planes/lines are additionally sampled at
        coarser resolutions (every `stride`-th texel) and all features are
        concatenated (multi-scale features).

        Returns:
            (C_total, N) tensor of plane * line features.
        """
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]],
                xyz_sampled[..., self.matMode[1]],
                xyz_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for stride in strides:
            for idx_plane in range(len(planes)):
                plane = planes[idx_plane]
                line = lines[idx_plane]
                if stride > 1:
                    plane = plane[:, :, ::stride, ::stride]
                    line = line[:, :, ::stride]
                plane_coef_point.append(
                    F.grid_sample(
                        plane, coordinate_plane[[idx_plane]], align_corners=True
                    ).view(-1, *xyz_sampled.shape[:1])
                )
                line_coef_point.append(
                    F.grid_sample(
                        line, coordinate_line[[idx_plane]], align_corners=True
                    ).view(-1, *xyz_sampled.shape[:1])
                )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )
        return plane_coef_point * line_coef_point

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        return plane_coef, line_coef

    def tv_loss(self, planes, lines, reg):
        total = 0
        for idx in range(len(planes)):
            total = total + reg(planes[idx]) * 1e-2 + reg(lines[idx]) * 1e-3
        return total

    def l1_density(self, planes, lines):
        A = torch.permute(
            planes[0][..., None] @ lines[0][..., 0][:, :, None, None],
            (0, 1, 3, 2, 4),
        )
        B = torch.permute(
            planes[1][..., None] @ lines[1][..., 0][:, :, None, None],
            (0, 1, 3, 4, 2),
        )
        C = torch.permute(
            planes[2][..., None] @ lines[2][..., 0][:, :, None, None],
            (0, 1, 4, 3, 2),
        )
        return torch.mean(
            torch.abs(self.feature2density(torch.sum(torch.cat([A, B, C], 1), 1)))
        )

    def normalize_coord(self, xyz_sampled):
        return (
            xyz_sampled - self.aabb.to(xyz_sampled.device)[0]
        ) * self.invaabbSize.to(xyz_sampled.device) - 1

    def unnormalize_coord(self, xyz_sampled):
        return (xyz_sampled + 1) / self.invaabbSize.to(
            xyz_sampled.device
        ) + self.aabb.to(xyz_sampled.device)[0]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def get_optparam_groups(self, lr_init_spatialxyz, lr_init_network):
        raise NotImplementedError

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "tSize": self.tSize,
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def save(self, se3_poses, focal_ratio_refine, path):
        kwargs = self.get_kwargs()
        kwargs["se3_poses"] = se3_poses
        kwargs["focal_ratio_refine"] = focal_ratio_refine
        torch.save({"kwargs": kwargs, "state_dict": self.state_dict()}, path)

    def load(self, ckpt):
        self.load_state_dict(ckpt["state_dict"])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = (
            (self.aabb.to(rays_pts.device)[0] > rays_pts)
            | (rays_pts > self.aabb.to(rays_pts.device)[1])
        ).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_contracted(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        inner_N_samples = N_samples - N_samples // 2
        outer_N_samples = N_samples // 2
        # inner
        interpx_inner = (
            torch.linspace(near, 2.0, inner_N_samples + 1).unsqueeze(0).to(rays_o)
        )
        if is_train:
            interpx_inner[:, :-1] += (
                torch.rand_like(interpx_inner).to(rays_o)
                * ((2.0 - near) / inner_N_samples)
            )[:, :-1]
        interpx_inner = (interpx_inner[:, 1:] + interpx_inner[:, :-1]) * 0.5
        # outer
        rng = torch.arange(outer_N_samples + 1)[None].float()
        if is_train:
            rng[:, :-1] += (torch.rand_like(rng).to(rng))[:, :-1]
        rng = torch.flip(rng, [1])
        rng = (rng[:, 1:] + rng[:, :-1]) * 0.5
        interpx_outer = 1.0 / (
            1 / (far) + (1 / 2.0 - 1 / (far)) * rng / outer_N_samples
        ).to(rays_o.device)
        interpx = torch.cat((interpx_inner, interpx_outer), -1)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        rays_pts_norm, _ = torch.max(torch.abs(rays_pts.clone()), dim=-1)
        contract_mask = rays_pts_norm > 1.0
        rays_pts[contract_mask] = (2 - 1 / rays_pts_norm[contract_mask])[..., None] * (
            rays_pts[contract_mask] / rays_pts_norm[contract_mask][..., None]
        )

        mask_outbbox = torch.zeros_like(rays_pts[..., 0]) > 0
        return rays_pts, interpx, ~mask_outbbox

    def viewdirs_and_dists(self, rays_chunk, z_vals, sample_shape):
        """Normalized view directions (expanded to every sample) and scaled
        distances between consecutive samples."""
        viewdirs = rays_chunk[:, 3:6]
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
            dim=-1,
        )
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        dists = dists * rays_norm
        viewdirs = viewdirs / rays_norm
        viewdirs = viewdirs.view(-1, 1, 3).expand(sample_shape)
        return viewdirs, dists * self.distance_scale

    @staticmethod
    def tile_time(ts_chunk, n_samples):
        return torch.tile(torch.unsqueeze(ts_chunk, -1), (1, n_samples))
