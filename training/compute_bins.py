"""Compute bin edges and anchor points from the actual training density
distribution under the NEW clamp0.9 pipeline.

Writes per-count histogram and proposes bins + anchors data-driven."""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as io
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


DATA_DIRS = [
    './site_data', './drone_data', './hacrowd_data',
    './qnrf_converted', './jhu_converted', './nwpu_converted',
]
INPUT_SIZE = 224
REDUCTION = 8
GRID = INPUT_SIZE // REDUCTION
SCALE_MIN, SCALE_MAX = 0.5, 3.0
CLAMP_FRAC = 0.9


def load_points(mat_path):
    points = np.empty((0, 2))
    try:
        mat = io.loadmat(mat_path)
        if 'points' in mat:
            raw = mat['points']
        elif 'annPoints' in mat:
            raw = mat['annPoints']
        elif 'gt_point' in mat:
            raw = mat['gt_point']
        else:
            keys = [k for k in mat.keys() if not k.startswith('_')]
            raw = mat[keys[0]] if keys else None
        if raw is not None:
            raw = np.array(raw)
            if raw.dtype == 'object':
                nums = []
                for item in raw.flat:
                    if isinstance(item, np.ndarray):
                        nums.extend(item.flatten().tolist())
                    elif isinstance(item, (int, float)):
                        nums.append(float(item))
                if nums and len(nums) % 2 == 0:
                    points = np.array(nums).reshape(-1, 2).astype(float)
            else:
                while raw.ndim > 2:
                    raw = raw.squeeze()
                if raw.ndim == 2:
                    if raw.shape[1] == 2:
                        points = raw.astype(float)
                    elif raw.shape[0] == 2:
                        points = raw.T.astype(float)
    except Exception:
        pass
    return points


def collect_samples():
    out = []
    for root in DATA_DIRS:
        p = Path(root)
        img_dir, mat_dir = p / 'images', p / 'mats'
        if not img_dir.is_dir() or not mat_dir.is_dir():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                m = mat_dir / (f.stem + '.mat')
                if m.exists():
                    out.append((str(f), str(m), p.name))
    return out


def pipeline(image, points, rng):
    w, h = image.size
    s = rng.uniform(SCALE_MIN, SCALE_MAX)
    target = int(INPUT_SIZE * s)
    cap = int(CLAMP_FRAC * min(h, w))
    crop = max(1, min(target, cap))
    top = rng.randint(0, max(0, h - crop))
    left = rng.randint(0, max(0, w - crop))
    image = TF.crop(image, top, left, crop, crop)
    if len(points) > 0:
        pts = points.copy()
        pts[:, 0] -= left
        pts[:, 1] -= top
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < crop) & \
               (pts[:, 1] >= 0) & (pts[:, 1] < crop)
        pts = pts[mask]
    else:
        pts = points
    if len(pts) > 0:
        pts[:, 0] *= INPUT_SIZE / crop
        pts[:, 1] *= INPUT_SIZE / crop
    return pts


def rasterize(points):
    d = np.zeros((GRID, GRID), dtype=np.float32)
    for p in points:
        x = int(p[0] / REDUCTION)
        y = int(p[1] / REDUCTION)
        if 0 <= x < GRID and 0 <= y < GRID:
            d[y, x] += 1.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-samples', type=int, default=600)
    ap.add_argument('--num-crops', type=int, default=2, help='crops per image (matches num_crops=2)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = collect_samples()
    print(f'total pairs: {len(samples):,}')

    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s[2]].append(s)
    per_ds = max(args.num_samples // len(by_ds), 1)
    picked = []
    for tag, lst in by_ds.items():
        picked.extend(rng.sample(lst, min(per_ds, len(lst))))
    print(f'auditing {len(picked)} samples x {args.num_crops} crops\n')

    all_cells = []
    per_ds_cells = defaultdict(list)

    for img_path, mat_path, tag in tqdm(picked, desc='audit'):
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        pts = load_points(mat_path)
        for _ in range(args.num_crops):
            out_pts = pipeline(image, pts, rng)
            cells = rasterize(out_pts).flatten()
            all_cells.append(cells)
            per_ds_cells[tag].append(cells)

    agg = np.concatenate(all_cells)
    print(f'\n{"=" * 70}')
    print(f'EXACT PER-COUNT HISTOGRAM (aggregate, {len(agg):,} cells)')
    print(f'{"=" * 70}')
    counter = Counter(agg.astype(int).tolist())
    total = len(agg)
    cum = 0
    print(f'  {"count":>6} {"#cells":>12} {"%":>7} {"cum%":>7}')
    for c in sorted(counter.keys()):
        if c > 20:
            break
        n = counter[c]
        cum += n
        print(f'  {c:>6} {n:>12,} {100*n/total:>6.3f}% {100*cum/total:>6.3f}%')
    above_20 = sum(v for k, v in counter.items() if k > 20)
    if above_20:
        print(f'  {">20":>6} {above_20:>12,} {100*above_20/total:>6.3f}%')

    print(f'\n  max observed: {int(agg.max())}')
    print(f'  nonzero cells: {100*(agg>0).sum()/total:.2f}%')
    nz = agg[agg > 0]
    for p in [50, 90, 95, 99, 99.5, 99.9, 99.99]:
        print(f'  p{p}(nonzero) = {np.percentile(nz, p):.2f}')

    print(f'\n{"=" * 70}')
    print('PER-DATASET MAX')
    print(f'{"=" * 70}')
    for tag in sorted(per_ds_cells.keys()):
        arr = np.concatenate(per_ds_cells[tag])
        nz_d = arr[arr > 0]
        print(f'  {tag:<18}  max={int(arr.max()):>3}  '
              f'p99={np.percentile(nz_d, 99):>5.1f}  '
              f'p99.9={np.percentile(nz_d, 99.9):>5.1f}')

    # Propose three bin schemes, compute data-driven anchors as *mean count*
    # inside each bin (matches CLIP-EBC paper's "average" anchor convention).
    schemes = {
        'current (max=5)': [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))],
        'wider (max=10)':  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, float('inf'))],
        'widest (max=17)': [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 11), (12, 16), (17, float('inf'))],
    }

    print(f'\n{"=" * 70}')
    print('PROPOSED BIN SCHEMES (anchors = mean of cells in bin, post-clip)')
    print(f'{"=" * 70}')
    for name, bins in schemes.items():
        max_count = int(bins[-1][0]) if bins[-1][1] == float('inf') else int(bins[-1][1])
        # For widest scheme, treat (17, inf) with max_count = ceil(observed_max)
        if bins[-1][1] == float('inf'):
            max_count = max(int(agg.max()), int(bins[-1][0]) + 2)
        clipped = np.minimum(agg, max_count)
        anchors = []
        clip_frac = (agg > max_count).sum() / total
        for lo, hi in bins:
            hi_eff = max_count if hi == float('inf') else hi
            m = (clipped >= lo) & (clipped <= hi_eff)
            a = float(clipped[m].mean()) if m.any() else float((lo + hi_eff) / 2)
            anchors.append(round(a, 4))
        print(f'\n  {name}')
        print(f'    bins          = {bins}')
        print(f'    anchor_points = {anchors}')
        print(f'    max_count     = {max_count}')
        print(f'    clipped cells = {100*clip_frac:.4f}%  (of {total:,})')


if __name__ == '__main__':
    main()
