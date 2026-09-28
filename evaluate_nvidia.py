"""PSNR / SSIM / LPIPS on the NVIDIA Dynamic Scenes benchmark.

Novel-view evaluation: the first camera (v000) is fixed and the time changes.
Rendered frames are the `fix_view/{t:03d}.png` outputs of
`train.py --render_only 1 --render_path 1`.

    python evaluate_nvidia.py --gt_dir data/nvidia_eval/gt --pred_dir log/nvidia
"""
import argparse
import os

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

SCENES = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]


def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def calculate_metrics(args, scene, lpips_loss):
    psnr, ssim, lp = 0.0, 0.0, 0.0
    frames = range(args.num_frames)
    for t in frames:
        img_true = cv2.imread(os.path.join(args.gt_dir, scene, f"{t:03d}.png"))
        img = cv2.imread(
            os.path.join(args.pred_dir, args.pred_pattern.format(scene=scene, t=t))
        )
        psnr += cv2.PSNR(img_true, img)
        ssim += structural_similarity(img_true, img, channel_axis=-1, win_size=3)
        lp += lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()
    n = len(frames)
    return psnr / n, ssim / n, lp / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="<gt_dir>/<scene>/{t:03d}.png")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument(
        "--pred_pattern",
        default="{scene}/fix_view/{t:03d}.png",
        help="path of a rendered frame relative to --pred_dir",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--num_frames", type=int, default=11, help="frames t = 0 .. num_frames-1")
    args = parser.parse_args()

    lpips_loss = lpips.LPIPS(net="alex")
    results = []
    for scene in args.scenes:
        metrics = calculate_metrics(args, scene, lpips_loss)
        results.append(metrics)
        print(f"{scene:<12} PSNR {metrics[0]:.2f} / SSIM {metrics[1]:.4f} / LPIPS {metrics[2]:.3f}")
    mean = np.mean(np.asarray(results), 0)
    print(f"{'Avg.':<12} PSNR {mean[0]:.2f} / SSIM {mean[1]:.4f} / LPIPS {mean[2]:.3f}")
