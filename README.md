# ICSD-NeRF: Independent Canonical Spaces for Enhanced Dynamic Scene Modeling in Neural Radiance Fields

Official implementation of **ICSD-NeRF** (IEEE Transactions on Computational Imaging, 2025).

[[Paper (IEEE Xplore)]](https://ieeexplore.ieee.org/document/11319162)

ICSD-NeRF reconstructs a dynamic scene from a monocular video with a static field and a dynamic field. The dynamic field is split into a **geometry** field and a **color** field, and each one warps points into **its own canonical space**. Camera poses and focal length are estimated jointly with the scene, so COLMAP poses are not required.

> **Note.** Hyperparameter values are not included in this release. The corresponding entries in `configs/*.txt` are left blank, and training stops with a list of the missing entries until they are filled in. Pretrained checkpoints are not provided.

## Code structure

| Component | Code |
| --- | --- |
| Static field with a rigidity (blending) branch on multi-scale features | `models/tensoRF.py` · `StaticTensorVM` (`compute_blendingfeature`) |
| Dynamic **geometry** field, warped into its own canonical space | `models/tensoRF.py` · `DynamicGeometryTensorVM` (`compute_densityfeature`) |
| Dynamic **color** field, warped into its own canonical space | `models/tensoRF.py` · `DynamicColorTensorVM` (`compute_appfeature`) |
| Deformation MLP (separate time and coordinate encoders) | `models/tensoRF.py` · `_DeformableField.warp_coordinate` |
| Multi-scale VM (TensoRF) features, strides 1/2/4 | `models/tensorBase.py` · `TensorBase.vm_features`, `MULTISCALE_STRIDES` |
| Late-view color decoder of the dynamic field | `models/tensorBase.py` · `MLPRender_Fea_late_view` |
| Static/dynamic compositing and dynamicness map | `renderer.py` · `raw2outputs`, `render_composite` |
| Scene flow and induced optical flow / disparity | `DynamicGeometryTensorVM.get_forward_backward_scene_flow`, `renderer.py` · `induce_flow` |
| Joint camera pose and focal length estimation | `camera.py` · `pose_to_mtx`, `train.py` (camera pose estimation section) |
| Training losses | `train.py` · `reconstruction` |
| Coarse-to-fine grid upsampling | `upsample_volume_grid` of each field |
| Novel view / time rendering (fixed view, view+time, dolly, zoom, spiral), FPS and model size | `train.py` · `render_test`, `generate_path` |
| NDC (forward-facing) and contracted (unbounded) rays | `--ray_type ndc / contract` |

```
├── train.py              # training, rendering of novel views / times
├── opt.py                # options (hyperparameters have no default)
├── renderer.py           # compositing, evaluation, path rendering, induced flow
├── models/               # static, dynamic-geometry and dynamic-color fields
├── dataLoader/           # NVIDIA / DAVIS video loaders
├── configs/              # nvidia.txt, davis.txt
├── scripts/              # preprocessing: DPT depth, RAFT flow, motion masks
└── evaluate_nvidia.py    # PSNR / SSIM / LPIPS on the NVIDIA benchmark
```

## Installation

```bash
conda create -n icsd-nerf python=3.9
conda activate icsd-nerf
pip install torch torchvision  # choose the build for your CUDA version
pip install -r requirements.txt
```

## Data

A scene folder looks like this:

```
<scene>/
├── images/          # frames
├── disp/ | dpt/     # monocular disparity  (NVIDIA: disp/000.npy, DAVIS: dpt/00000.npy)
├── flow/            # {idx}_fwd.npz, {idx}_bwd.npz  (flow + consistency mask)
└── motion_masks/ | epipolar_error_png/   # motion masks (--use_foreground_mask)
```

**NVIDIA Dynamic Scenes.** Use the preprocessed data of [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF), which contains `images`, `disp`, `flow` and `motion_masks`.

**DAVIS / your own video.** Put the frames in `<scene>/images/` as `00000.png`, `00001.png`, … and run:

```bash
python scripts/generate_DPT.py  --dataset_path <scene> --model <path/to/dpt_large-midas-2f21e586.pt>
python scripts/generate_flow.py --dataset_path <scene> --model <path/to/raft-things.pth>
python scripts/generate_mask.py --dataset_path <scene>
```

The weights are available from [DPT](https://github.com/isl-org/DPT) and [RAFT](https://github.com/princeton-vl/RAFT).

## Training

Fill in the hyperparameter block of the config first. Then run:

```bash
python train.py --config configs/nvidia.txt --datadir data/nvidia/Balloon1 --expname Balloon1
python train.py --config configs/davis.txt  --datadir data/davis/train     --expname train
```

Checkpoints (`<expname>_static.th`, `<expname>_dynamic_g.th`, `<expname>_dynamic_c.th`), TensorBoard logs, the reconstructed video and the estimated cameras (`poses_bounds_estimated.npy`) are written to `<basedir>/<expname>/`.

## Rendering and evaluation

```bash
# reconstruction + novel view / time paths (fix_view, change_view_time, dolly, zoom, spiral)
python train.py --config configs/nvidia.txt --datadir data/nvidia/Balloon1 --expname Balloon1 \
    --ckpt log/nvidia/Balloon1/Balloon1 --render_only 1 --render_test 1 --render_path 1

# NVIDIA benchmark (fixed camera, changing time)
python evaluate_nvidia.py --gt_dir <gt_dir> --pred_dir log/nvidia
```

`evaluate_nvidia.py` reads the ground-truth frames of the evaluation camera from `<gt_dir>/<scene>/{t:03d}.png`.

## Acknowledgements

This code is built on [RoDynRF](https://github.com/facebookresearch/robust-dynrf) and [TensoRF](https://github.com/apchenstu/TensoRF). The preprocessing uses [RAFT](https://github.com/princeton-vl/RAFT) and [DPT](https://github.com/isl-org/DPT), and the evaluation protocol follows [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF). We thank the authors for releasing their code.

## Citation

```bibtex
@article{yeom2025icsdnerf,
  title   = {{ICSD-NeRF}: Independent Canonical Spaces for Enhanced Dynamic Scene Modeling in Neural Radiance Fields},
  author  = {Yeom, S. and Son, H. and Kang, C. and Shin, E. and Kim, J. and Yun, K. J. and Kang, S.-J.},
  journal = {IEEE Transactions on Computational Imaging},
  year    = {2025}
}
```

## License

MIT License (see [LICENSE](LICENSE)). Parts of the code are derived from RoDynRF (MIT, Copyright (c) Meta Platforms, Inc.).
