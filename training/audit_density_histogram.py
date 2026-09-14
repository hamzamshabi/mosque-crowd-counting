"""Audit per-cell density distribution under the current whole-image-resize
pipeline vs the proposed RandomResizedCrop pipeline.

Now with per-dataset breakdown and fallback-rate logging across multiple scale
ranges, plus a recommended-bins decision.

Usage:
    python audit_density_histogram.py --num-samples 300
"""

import argparse
import random
from collections import defaultdict
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

SCALE_RANGES = [(0.75, 1.5), (0.5, 2.0), (0.5, 2.5), (0.5, 3.0)]
CLAMPED_RANGES = [(0.5, 3.0, 0.9)]  # (lo, hi, max_fraction_of_short_side)


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
                    elif raw.shape[0] == 1 and raw.shape[1] > 2:
                        flat = raw.flatten()
                        if flat.size % 2 == 0 and flat.size > 0:
                            points = flat.reshape(-1, 2).astype(float)
                elif raw.ndim == 1:
                    if raw.size % 2 == 0 and raw.size >= 2:
                        points = raw.reshape(-1, 2).astype(float)
    except Exception as exc:
        print(f"WARNING: failed to load {mat_path}: {exc}")
    return points


def collect_samples():
    """Return list of (img_path, mat_path, dataset_tag)."""
    samples = []
    for root in DATA_DIRS:
        root_p = Path(root)
        img_dir = root_p / 'images'
        mat_dir = root_p / 'mats'
        tag = root_p.name
        if not img_dir.is_dir() or not mat_dir.is_dir():
            print(f"  skip {root_p} (missing images/ or mats/)")
            continue
        count = 0
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                mat_path = mat_dir / (f.stem + '.mat')
                if mat_path.exists():
                    samples.append((str(f), str(mat_path), tag))
                    count += 1
        print(f"  {tag}: {count} pairs")
    return samples


def pipeline_current(image, points, rng):
    w, h = image.size
    scale = rng.uniform(0.7, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    left = rng.randint(0, max(0, w - crop_w))
    top = rng.randint(0, max(0, h - crop_h))
    image = TF.crop(image, top, left, crop_h, crop_w)
    if len(points) > 0:
        pts = points.copy()
        pts[:, 0] -= left
        pts[:, 1] -= top
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < crop_w) & (pts[:, 1] >= 0) & (pts[:, 1] < crop_h)
        pts = pts[mask]
    else:
        pts = points
    image = TF.resize(image, (INPUT_SIZE, INPUT_SIZE))
    if len(pts) > 0:
        pts[:, 0] *= INPUT_SIZE / crop_w
        pts[:, 1] *= INPUT_SIZE / crop_h
    return image, pts, False


def pipeline_new(image, points, rng, scale_range, clamp_frac=None):
    """Returns (image, pts, fallback_used). When clamp_frac is set, crop size is
    capped at clamp_frac * min(h, w) so we never exceed that fraction of the
    native short side -> no upsample fallback."""
    w, h = image.size
    s = rng.uniform(*scale_range)
    crop_h = int(INPUT_SIZE * s)
    crop_w = int(INPUT_SIZE * s)
    if clamp_frac is not None:
        cap = int(clamp_frac * min(h, w))
        crop_h = min(crop_h, cap)
        crop_w = min(crop_w, cap)
    fallback = crop_h > h or crop_w > w
    if fallback:
        ratio = max(crop_h / h, crop_w / w)
        new_h = int(h * ratio) + 1
        new_w = int(w * ratio) + 1
        image = TF.resize(image, (new_h, new_w))
        if len(points) > 0:
            pts = points.copy()
            pts[:, 0] *= new_w / w
            pts[:, 1] *= new_h / h
        else:
            pts = points
        w, h = new_w, new_h
    else:
        pts = points.copy() if len(points) else points
    top = rng.randint(0, max(0, h - crop_h))
    left = rng.randint(0, max(0, w - crop_w))
    image = TF.crop(image, top, left, crop_h, crop_w)
    if len(pts) > 0:
        pts[:, 0] -= left
        pts[:, 1] -= top
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < crop_w) & (pts[:, 1] >= 0) & (pts[:, 1] < crop_h)
        pts = pts[mask]
    image = TF.resize(image, (INPUT_SIZE, INPUT_SIZE))
    if len(pts) > 0:
        pts[:, 0] *= INPUT_SIZE / crop_w
        pts[:, 1] *= INPUT_SIZE / crop_h
    return image, pts, fallback


def rasterize(points):
    d = np.zeros((GRID, GRID), dtype=np.float32)
    for p in points:
        x = int(p[0] / REDUCTION)
        y = int(p[1] / REDUCTION)
        if 0 <= x < GRID and 0 <= y < GRID:
            d[y, x] += 1.0
    return d


def describe(cells, label, fallback_rate=None):
    cells = np.asarray(cells)
    total = len(cells)
    if total == 0:
        print(f"  [{label}] no cells")
        return None
    nonzero = cells[cells > 0]
    nz_pct = 100 * len(nonzero) / total
    fb_str = f"  fallback={fallback_rate*100:.1f}%" if fallback_rate is not None else ""
    if len(nonzero) == 0:
        print(f"  [{label}] {total:,} cells, 0 nonzero{fb_str}")
        return {'p99': 0.0, 'max': 0.0, 'fallback': fallback_rate}
    p50 = np.percentile(nonzero, 50)
    p90 = np.percentile(nonzero, 90)
    p99 = np.percentile(nonzero, 99)
    p999 = np.percentile(nonzero, 99.9)
    mx = nonzero.max()
    print(f"  [{label}] nonzero={nz_pct:5.2f}%  max={mx:5.0f}  "
          f"p50={p50:4.1f} p90={p90:4.1f} p99={p99:5.1f} p99.9={p999:5.1f}  "
          f"mean_all={cells.mean():.3f}{fb_str}")
    return {'p99': float(p99), 'max': float(mx), 'fallback': fallback_rate}


def recommend_bins(p99, cells):
    if p99 <= 3:
        edges = [(0, 0), (1, 1), (2, 2), (3, float('inf'))]
        max_count = 3
    elif p99 <= 6:
        edges = [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))]
        max_count = 5
    elif p99 <= 12:
        edges = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, float('inf'))]
        max_count = 10
    else:
        edges = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 11), (12, 16), (17, float('inf'))]
        max_count = 17

    cells_clipped = np.minimum(cells, max_count)
    anchors = []
    for lo, hi in edges:
        hi_eff = max_count if hi == float('inf') else hi
        m = (cells_clipped >= lo) & (cells_clipped <= hi_eff)
        if m.any():
            anchors.append(float(cells_clipped[m].mean()))
        else:
            anchors.append(float((lo + hi_eff) / 2))
    return edges, anchors, max_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-samples', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    samples = collect_samples()
    if not samples:
        raise SystemExit("no (image, mat) pairs found")
    print(f"\ntotal pairs: {len(samples):,}")

    # stratified sample: up to N/num_datasets per dataset
    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s[2]].append(s)
    per_ds = max(args.num_samples // max(len(by_ds), 1), 1)
    picked = []
    for tag, lst in by_ds.items():
        k = min(per_ds, len(lst))
        picked.extend(rng.sample(lst, k))
    print(f"auditing {len(picked)} samples ({per_ds}/dataset, seed={args.seed})\n")

    # storage: cells_by[(scheme_label)][dataset_tag] = list of arrays
    # fallback_by[scheme_label][dataset_tag] = (hits, total)
    cells_by = defaultdict(lambda: defaultdict(list))
    fallback_by = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    schemes = [('CURRENT', None, None)]
    schemes += [(f'NEW[{lo},{hi}]', (lo, hi), None) for lo, hi in SCALE_RANGES]
    schemes += [(f'NEW[{lo},{hi}]_clamp{cf}', (lo, hi), cf) for lo, hi, cf in CLAMPED_RANGES]

    for img_path, mat_path, tag in tqdm(picked, desc='audit'):
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as exc:
            print(f"skip {img_path}: {exc}")
            continue
        pts = load_points(mat_path)

        for label, rng_scale, clamp in schemes:
            if rng_scale is None:
                _, pts_out, fb = pipeline_current(image, pts, rng)
            else:
                _, pts_out, fb = pipeline_new(image, pts, rng, rng_scale, clamp_frac=clamp)
            cells_by[label][tag].append(rasterize(pts_out).flatten())
            fallback_by[label][tag][0] += int(fb)
            fallback_by[label][tag][1] += 1

    # Report
    for label, _, _ in schemes:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        all_cells = []
        all_fb = [0, 0]
        for tag in sorted(cells_by[label].keys()):
            arr = np.concatenate(cells_by[label][tag])
            all_cells.append(arr)
            fb_hits, fb_total = fallback_by[label][tag]
            all_fb[0] += fb_hits
            all_fb[1] += fb_total
            fb_rate = fb_hits / fb_total if fb_total else 0.0
            describe(arr, f"{tag:<18}", fallback_rate=fb_rate)
        agg = np.concatenate(all_cells) if all_cells else np.array([])
        fb_rate_all = all_fb[0] / all_fb[1] if all_fb[1] else 0.0
        agg_stats = describe(agg, f"{'AGGREGATE':<18}", fallback_rate=fb_rate_all)

        if label.startswith('NEW') and agg_stats is not None:
            edges, anchors, max_count = recommend_bins(agg_stats['p99'], agg)
            print(f"  -> suggested bins: {edges}")
            print(f"  -> suggested anchor_points: {[round(a,4) for a in anchors]}")
            print(f"  -> suggested max_count: {max_count}")

    # Drone-focused summary
    print(f"\n{'='*70}\nDRONE-FOCUSED SUMMARY (drone_data only)\n{'='*70}")
    print(f"{'scheme':<26}  {'p99':>6}  {'max':>5}  {'fallback':>9}  {'nz%':>6}")
    for label, _, _ in schemes:
        if 'drone_data' not in cells_by[label]:
            continue
        arr = np.concatenate(cells_by[label]['drone_data'])
        nz = arr[arr > 0]
        fb_hits, fb_total = fallback_by[label]['drone_data']
        fb = fb_hits / fb_total if fb_total else 0.0
        p99 = np.percentile(nz, 99) if len(nz) else 0.0
        mx = nz.max() if len(nz) else 0.0
        nz_pct = 100 * len(nz) / len(arr) if len(arr) else 0.0
        print(f"{label:<26}  {p99:>6.1f}  {mx:>5.0f}  {fb*100:>8.1f}%  {nz_pct:>5.2f}%")

    print("\nDecision rule: pick the NEW range where drone p99 >= 4 AND")
    print("no dataset fallback > 20%. Then take the suggested bins for that range.")


if __name__ == "__main__":
    main()
