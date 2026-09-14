"""
Run CLIP-EBC checkpoints on one image and save a density-map comparison figure.

Uses the same native-resolution protocol as eval/test_qnrf.py: the image is
zero-padded to a multiple of 224, tiled into non-overlapping 224x224 windows,
and the predicted densities are summed. No whole-image resize.

Example:
    python demo/predict_image.py --clip-ebc ../CLIP-EBC \
        --checkpoint checkpoints/best_model.pth \
        --baseline ../CLIP-EBC/best_mae.pth \
        --image UCF-QNRF_ECCV18/Test/img_0104.jpg \
        --out docs/images/img_0104.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

WINDOW = 224
DISPLAY_MAX_SIDE = 1200
DEFAULT_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, float("inf"))]
DEFAULT_ANCHORS = [0, 1, 2, 3, 4.21937]
CLIP_ARGS = dict(prompt_type="word", num_vpt=32, deep_vpt=True, vpt_drop=0.0)
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_model(path, device, lora_rank=16):
    """Fine-tuned checkpoints carry bins/anchors and LoRA weights; the
    original pretrained file is a raw state dict with the default 5 bins."""
    from models import get_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    fine_tuned = isinstance(ckpt, dict) and "model_state_dict" in ckpt
    bins = ckpt.get("bins", DEFAULT_BINS) if fine_tuned else DEFAULT_BINS
    anchors = ckpt.get("anchor_points", DEFAULT_ANCHORS) if fine_tuned else DEFAULT_ANCHORS

    model = get_model(backbone="clip_vit_b_16", input_size=WINDOW, reduction=8,
                      bins=bins, anchor_points=anchors, **CLIP_ARGS)
    if fine_tuned:
        from lora_utils import inject_lora_into_clip_vit
        model = inject_lora_into_clip_vit(model, rank=lora_rank)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image, device):
    """Return (density map cropped to the image, total count)."""
    x = NORMALIZE(TF.to_tensor(image)).unsqueeze(0).to(device)
    _, _, h, w = x.shape
    H = -(-h // WINDOW) * WINDOW
    W = -(-w // WINDOW) * WINDOW
    x = F.pad(x, (0, W - w, 0, H - h), value=0.0)

    rows = []
    for top in range(0, H, WINDOW):
        row = []
        for left in range(0, W, WINDOW):
            out = model(x[:, :, top:top + WINDOW, left:left + WINDOW])
            if isinstance(out, tuple):
                out = out[1]
            row.append(out[0, 0].cpu().numpy())
        rows.append(np.concatenate(row, axis=1))
    density = np.concatenate(rows, axis=0)

    count = float(density.sum())
    scale = density.shape[0] / H
    density = density[:max(1, round(h * scale)), :max(1, round(w * scale))]
    return density, count


def ground_truth(image_path):
    """UCF-QNRF style: img_0001.jpg next to img_0001_ann.mat with annPoints."""
    mat_path = os.path.splitext(image_path)[0] + "_ann.mat"
    if not os.path.exists(mat_path):
        return None
    return len(sio.loadmat(mat_path)["annPoints"])


def save_figure(image, panels, gt, out_path):
    """panels: list of (title, density, count)."""
    scale = min(1.0, DISPLAY_MAX_SIDE / max(image.size))
    shown = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    maps = [np.array(Image.fromarray(d.astype(np.float32)).resize(shown.size, Image.BILINEAR))
            for _, d, _ in panels]
    vmax = max(np.percentile(m, 99.5) for m in maps) or 1.0

    n = len(panels) + 1
    aspect = shown.width / shown.height
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6 / aspect + 0.9), dpi=130)

    axes[0].imshow(shown)
    axes[0].set_title("Input" + (f"  |  ground truth: {gt:,}" if gt is not None else ""), fontsize=13)
    for ax, (title, _, count), m in zip(axes[1:], panels, maps):
        ax.imshow(shown)
        ax.imshow(m, cmap="jet", alpha=0.55, vmin=0, vmax=vmax)
        label = f"{title}\npredicted: {count:,.0f}"
        if gt is not None:
            label += f"  ({count - gt:+,.0f})"
        ax.set_title(label, fontsize=13)
    for ax in axes:
        ax.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip-ebc", required=True, help="path to a CLIP-EBC checkout")
    parser.add_argument("--checkpoint", required=True, help="fine-tuned LoRA checkpoint")
    parser.add_argument("--baseline", help="original pretrained CLIP-EBC weights (optional middle panel)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.clip_ebc))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

    image = Image.open(args.image).convert("RGB")
    gt = ground_truth(args.image)
    panels = []

    if args.baseline:
        density, count = predict(load_model(args.baseline, args.device), image, args.device)
        print(f"baseline:   {count:.2f}")
        panels.append(("Pretrained CLIP-EBC", density, count))

    density, count = predict(load_model(args.checkpoint, args.device), image, args.device)
    print(f"fine-tuned: {count:.2f}")
    panels.append(("LoRA fine-tuned (this repo)", density, count))

    if gt is not None:
        print(f"ground truth: {gt}")
    save_figure(image, panels, gt, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
