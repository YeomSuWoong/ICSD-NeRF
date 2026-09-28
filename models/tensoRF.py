# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The three radiance fields of ICSD-NeRF.

- StaticTensorVM: static scene (density, color) + rigidity (blending) field.
- DynamicGeometryTensorVM: time-varying density with its own deformation MLP,
  plus the forward/backward scene-flow MLP.
- DynamicColorTensorVM: time-varying color with its own deformation MLP.

The dynamic scene is thus represented by separate geometry and color fields,
each sampled in its own canonical space.
"""

import torch
import torch.nn.functional as F

from .tensorBase import TensorBase, positional_encoding, raw2alpha

# Strides at which the dynamic / rigidity VM planes are sampled (multi-scale features).
MULTISCALE_STRIDES = (1, 2, 4)


class StaticTensorVM(TensorBase):
    """Static field with a rigidity (blending) branch.

    The blending value predicted at every sample decides how much the static
    and the dynamic field contribute to the composited rendering.
    """

    def __init__(self, aabb, gridSize, tSize, device, **kargs):
        super(StaticTensorVM, self).__init__(aabb, gridSize, tSize, device, **kargs)

        self.density_layer1 = torch.nn.Linear(
            sum(self.density_n_comp) + 3 + 10 * 2 * 3, 32
        ).to(device)
        self.density_layer2 = torch.nn.Linear(32, 1).to(device)

        self.blending_layer1 = torch.nn.Linear(
            sum(self.density_n_comp) * len(MULTISCALE_STRIDES) + 3 + 10 * 2 * 3, 64
        ).to(device)
        self.blending_layer2 = torch.nn.Linear(64, 1).to(device)

    def init_svd_volume(self, res, device):
        self.blending_plane, self.blending_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz, lr_init_network):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.blending_line, "lr": lr_init_spatialxyz},
            {"params": self.blending_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
            {"params": self.density_layer1.parameters(), "lr": lr_init_network},
            {"params": self.density_layer2.parameters(), "lr": lr_init_network},
            {"params": self.renderModule.parameters(), "lr": lr_init_network},
        ]
        return grad_vars

    def density_L1(self):
        return self.l1_density(self.density_plane, self.density_line)

    def TV_loss_blending(self, reg):
        return self.tv_loss(self.blending_plane, self.blending_line, reg)

    def TV_loss_density(self, reg):
        return self.tv_loss(self.density_plane, self.density_line, reg)

    def TV_loss_app(self, reg):
        return self.tv_loss(self.app_plane, self.app_line, reg)

    def compute_densityfeature(self, xyz_sampled):
        indata = [self.vm_features(self.density_plane, self.density_line, xyz_sampled).T]
        indata += [xyz_sampled]
        indata += [positional_encoding(xyz_sampled, 10)]

        sigma_feature = self.density_layer2(
            F.relu(self.density_layer1(torch.cat(indata, -1)), inplace=True)
        )
        return sigma_feature[..., 0]

    def compute_appfeature(self, xyz_sampled):
        return self.basis_mat(
            self.vm_features(self.app_plane, self.app_line, xyz_sampled).T
        )

    def compute_blendingfeature(self, xyz_sampled):
        """Rigidity (blending) logit from multi-scale VM features."""
        indata = [
            self.vm_features(
                self.blending_plane,
                self.blending_line,
                xyz_sampled,
                strides=MULTISCALE_STRIDES,
            ).T
        ]
        indata += [xyz_sampled]
        indata += [positional_encoding(xyz_sampled, 10)]

        blending_feature = self.blending_layer2(
            F.relu(self.blending_layer1(torch.cat(indata, -1)), inplace=True)
        )
        return blending_feature[..., 0]

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )
        self.blending_plane, self.blending_line = self.up_sampling_VM(
            self.blending_plane, self.blending_line, res_target
        )
        self.update_stepSize(res_target)

    def forward(self, rays_chunk, ts_chunk, xyz_sampled, z_vals, ray_valid):
        """
        Returns:
            blending: (N_rays, N_samples, 1) rigidity/blending in [0, 1]
            weight:   (N_rays, N_samples) volume-rendering weights of the static field
            rgb:      (N_rays, N_samples, 3)
            sigma:    (N_rays, N_samples)
        """
        viewdirs, dists = self.viewdirs_and_dists(rays_chunk, z_vals, xyz_sampled.shape)
        t_sampled = self.tile_time(ts_chunk, xyz_sampled.shape[1])

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        blending = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)

        normalized_xyz_sampled = self.normalize_coord(xyz_sampled)
        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(
                normalized_xyz_sampled[ray_valid]
            )
            sigma[ray_valid] = self.feature2density(sigma_feature)

        _, weight, _ = raw2alpha(sigma, dists)

        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            app_features = self.compute_appfeature(normalized_xyz_sampled[app_mask])
            rgb[app_mask] = self.renderModule(
                normalized_xyz_sampled[app_mask],
                viewdirs[app_mask],
                app_features,
                t_sampled[app_mask][..., None],
            )

        blending_features = self.compute_blendingfeature(
            normalized_xyz_sampled[ray_valid]
        )
        blending[ray_valid] = torch.sigmoid(blending_features)[:, None]

        return blending, weight, rgb, sigma


class _DeformableField:
    """Deformation MLP mapping a point x at time t to its canonical position x'.

    A time branch and a coordinate branch are encoded separately and fused to
    predict the offset: x' = x + D(enc(x), enc(t)).
    """

    def build_deformation_mlp(self, device):
        self.layer1 = torch.nn.Linear(1 + 8 * 2 * 1, 64).to(device)
        self.layer2 = torch.nn.Linear(64, 30).to(device)
        self.layer3 = torch.nn.Linear(3 + 10 * 2 * 3, 128).to(device)
        self.layer4 = torch.nn.Linear(128, 60).to(device)
        self.layer5 = torch.nn.Linear(30 + 60, 64).to(device)
        self.layer6 = torch.nn.Linear(64, 64).to(device)
        self.layer7 = torch.nn.Linear(64, 3).to(device)

    def deformation_parameters(self):
        return [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
        ]

    def warp_coordinate(self, unnormalized_xyz_sampled, t_sampled):
        indata = [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 8)]
        t_out = self.layer2(F.relu(self.layer1(torch.cat(indata, dim=-1)), inplace=True))

        normalized_xyz = self.normalize_coord(unnormalized_xyz_sampled)
        indata_xyz = [normalized_xyz]
        indata_xyz += [positional_encoding(normalized_xyz, 10)]
        coord_out = self.layer4(
            F.relu(self.layer3(torch.cat(indata_xyz, dim=-1)), inplace=True)
        )

        delta_xyz = self.layer7(
            F.relu(
                self.layer6(
                    F.relu(self.layer5(torch.cat([coord_out, t_out], dim=-1)), inplace=True)
                ),
                inplace=True,
            )
        )
        return unnormalized_xyz_sampled + delta_xyz

    def canonical_coord(self, xyz_sampled, t_sampled):
        """Normalized point -> normalized canonical point."""
        return self.normalize_coord(
            self.warp_coordinate(self.unnormalize_coord(xyz_sampled), t_sampled)
        )


class DynamicColorTensorVM(_DeformableField, TensorBase):
    """Dynamic color field (its own deformation + multi-scale appearance planes)."""

    def __init__(self, aabb, gridSize, tSize, device, **kargs):
        TensorBase.__init__(self, aabb, gridSize, tSize, device, **kargs)
        self.build_deformation_mlp(device)

    def init_svd_volume(self, res, device):
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp) * len(MULTISCALE_STRIDES), self.app_dim, bias=False
        ).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz, lr_init_network):
        grad_vars = [
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        grad_vars += [
            {"params": layer.parameters(), "lr": lr_init_network}
            for layer in self.deformation_parameters()
        ]
        grad_vars += [{"params": self.renderModule.parameters(), "lr": lr_init_network}]
        return grad_vars

    def TV_loss_app(self, reg):
        return self.tv_loss(self.app_plane, self.app_line, reg)

    def compute_appfeature(self, xyz_sampled, t_sampled):
        xyz_prime_sampled = self.canonical_coord(xyz_sampled, t_sampled)
        features = self.vm_features(
            self.app_plane, self.app_line, xyz_prime_sampled, strides=MULTISCALE_STRIDES
        )
        return self.basis_mat(features.T), xyz_prime_sampled

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.update_stepSize(res_target)

    def forward(self, rays_chunk, ts_chunk, xyz_sampled, z_vals, ray_valid):
        """Returns rgb: (N_rays, N_samples, 3)."""
        viewdirs, _ = self.viewdirs_and_dists(rays_chunk, z_vals, xyz_sampled.shape)
        t_sampled = self.tile_time(ts_chunk, xyz_sampled.shape[1])

        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        if ray_valid.any():
            normalized_xyz_sampled = self.normalize_coord(xyz_sampled)
            app_features, xyz_prime_sampled = self.compute_appfeature(
                normalized_xyz_sampled[ray_valid], t_sampled[ray_valid]
            )
            rgb[ray_valid] = self.renderModule(
                xyz_prime_sampled,
                viewdirs[ray_valid],
                app_features,
                t_sampled[ray_valid][..., None],
            )
        return rgb


class DynamicGeometryTensorVM(_DeformableField, TensorBase):
    """Dynamic geometry field (its own deformation + multi-scale density planes)
    and the forward/backward scene-flow MLP."""

    def __init__(self, aabb, gridSize, tSize, device, **kargs):
        TensorBase.__init__(self, aabb, gridSize, tSize, device, **kargs)
        self.build_deformation_mlp(device)

        self.density_layer1 = torch.nn.Linear(
            sum(self.density_n_comp) * len(MULTISCALE_STRIDES)
            + 3
            + 10 * 2 * 3
            + 1
            + 8 * 2 * 1,
            64,
        ).to(device)
        self.density_layer2 = torch.nn.Linear(64, 1).to(device)

        layer_sf_1 = torch.nn.Linear(4 * 2 * 4 + 4, 64)  # PE of (x, y, z, t)
        layer_sf_2 = torch.nn.Linear(64, 64)
        layer_sf_3 = torch.nn.Linear(64, 64)
        layer_sf_4 = torch.nn.Linear(64, 6)  # forward (3) + backward (3) flow
        self.scene_flow_mlp = torch.nn.Sequential(
            layer_sf_1,
            torch.nn.ReLU(inplace=True),
            layer_sf_2,
            torch.nn.ReLU(inplace=True),
            layer_sf_3,
            torch.nn.ReLU(inplace=True),
            layer_sf_4,
        ).to(device)

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )

    def get_optparam_groups(self, lr_init_spatialxyz, lr_init_network):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.scene_flow_mlp.parameters(), "lr": lr_init_network},
        ]
        grad_vars += [
            {"params": layer.parameters(), "lr": lr_init_network}
            for layer in self.deformation_parameters()
        ]
        grad_vars += [
            {"params": self.density_layer1.parameters(), "lr": lr_init_network},
            {"params": self.density_layer2.parameters(), "lr": lr_init_network},
            {"params": self.renderModule.parameters(), "lr": lr_init_network},
        ]
        return grad_vars

    def density_L1(self):
        return self.l1_density(self.density_plane, self.density_line)

    def TV_loss_density(self, reg):
        return self.tv_loss(self.density_plane, self.density_line, reg)

    def get_forward_backward_scene_flow(self, unnormalized_pts, t_sampled):
        flattened_unnormalized_pts = unnormalized_pts.view(-1, 3)
        flattened_t_sampled = (
            (t_sampled[:, None]).repeat(1, unnormalized_pts.shape[1]).view(-1, 1)
        )
        indata = [self.normalize_coord(flattened_unnormalized_pts)]
        indata += [
            positional_encoding(self.normalize_coord(flattened_unnormalized_pts), 4)
        ]
        indata += [flattened_t_sampled]
        indata += [positional_encoding(flattened_t_sampled, 4)]
        scene_flow = self.scene_flow_mlp(torch.cat(indata, -1)).view(
            unnormalized_pts.shape[0], unnormalized_pts.shape[1], 6
        )
        scene_flow_f = scene_flow[..., 0:3]
        scene_flow_b = scene_flow[..., 3:6]
        return scene_flow_f, scene_flow_b

    def compute_densityfeature(self, xyz_sampled, t_sampled):
        xyz_prime_sampled = self.canonical_coord(xyz_sampled, t_sampled)
        features = self.vm_features(
            self.density_plane,
            self.density_line,
            xyz_prime_sampled,
            strides=MULTISCALE_STRIDES,
        )

        indata = [features.T]
        indata += [xyz_prime_sampled]
        indata += [positional_encoding(xyz_prime_sampled, 10)]
        indata += [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 8)]

        sigma_feature = self.density_layer2(
            F.relu(self.density_layer1(torch.cat(indata, -1)), inplace=True)
        )
        return sigma_feature[..., 0]

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )
        self.update_stepSize(res_target)

    def forward(self, rays_chunk, ts_chunk, xyz_sampled, z_vals, ray_valid):
        """
        Returns:
            sigma: (N_rays, N_samples)
            dists: (N_rays, N_samples) scaled sample distances (shared by all fields)
        """
        _, dists = self.viewdirs_and_dists(rays_chunk, z_vals, xyz_sampled.shape)
        t_sampled = self.tile_time(ts_chunk, xyz_sampled.shape[1])

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        if ray_valid.any():
            normalized_xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(
                normalized_xyz_sampled[ray_valid], t_sampled[ray_valid]
            )
            sigma[ray_valid] = self.feature2density(sigma_feature)
        return sigma, dists
