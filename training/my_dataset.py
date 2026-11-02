import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
import scipy.io as io
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class CrowdDataset(Dataset):
    def __init__(self, root_dirs, reduction=8, input_size=224, max_count=5, augment=True,
                 num_crops=1, scale_min=0.5, scale_max=3.0, max_crop_frac=0.9):
        """
        Multi-source crowd counting dataset (mosque, drone, and public benchmarks).

        Training path: clamped RandomResizedCrop at NATIVE pixel scale, then
        resize to input_size. This keeps per-cell densities aligned with what
        sliding-window inference sees at native resolution.

        Args:
            root_dirs: Path or list of paths. Each must contain 'images/' and 'mats/' subdirs.
            reduction: Reduction factor for density map (default: 8)
            input_size: Input image size (default: 224)
            max_count: Maximum count per grid cell (label correction)
            augment: Enable data augmentation (training path)
            num_crops: Number of augmented crops per image per epoch
            scale_min / scale_max: crop scale range at native pixels. Crop side is
                int(input_size * s) where s ~ U(scale_min, scale_max).
            max_crop_frac: upper bound on crop side as a fraction of min(h, w) —
                prevents upsample fallback on small images. Audit picked 0.9.
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.reduction = reduction
        self.max_count = max_count
        self.input_size = input_size
        self.augment = augment
        self.num_crops = num_crops
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_crop_frac = max_crop_frac

        # Collect (img_path, mat_path, dataset_tag) from all root directories.
        # dataset_tag is used by the training script's WeightedRandomSampler to
        # oversample drone_data.
        self.samples = []
        for root in root_dirs:
            tag = os.path.basename(os.path.normpath(root))
            img_dir = os.path.join(root, 'images')
            mat_dir = os.path.join(root, 'mats')
            if not os.path.isdir(img_dir) or not os.path.isdir(mat_dir):
                print(f"WARNING: {root} missing images/ or mats/ subdir, skipping.")
                continue
            for f in sorted(os.listdir(img_dir)):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    mat_name = os.path.splitext(f)[0] + '.mat'
                    mat_path = os.path.join(mat_dir, mat_name)
                    if os.path.exists(mat_path):
                        self.samples.append((os.path.join(img_dir, f), mat_path, tag))
                    else:
                        print(f"WARNING: no mat for {f}, skipping.")

        # ImageNet normalization (matches official CLIP-EBC datasets/crowd.py)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0  # match author
        )

        print(f"Loaded {len(self.samples)} samples from {root_dirs}")
        print(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
        print(f"Multi-crop: {num_crops} crops per image")

    def __len__(self):
        return len(self.samples)

    def _clamped_random_resized_crop(self, image, points):
        """Clamped RandomResizedCrop at native pixel scale.

        Samples s ~ U(scale_min, scale_max), computes target crop side
        int(input_size * s), caps it at max_crop_frac * min(h, w), then random-
        crops a square at native resolution. Points are translated to crop-local
        coords; caller is responsible for the final resize to input_size.
        Returns (cropped_image, cropped_points, crop_size).
        """
        w, h = image.size
        s = random.uniform(self.scale_min, self.scale_max)
        target = int(self.input_size * s)
        cap = max(1, int(self.max_crop_frac * min(h, w)))
        crop_size = max(1, min(target, cap))

        top = random.randint(0, max(0, h - crop_size))
        left = random.randint(0, max(0, w - crop_size))
        image = TF.crop(image, top, left, crop_size, crop_size)

        if len(points) > 0:
            pts = points.copy()
            pts[:, 0] -= left
            pts[:, 1] -= top
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < crop_size) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < crop_size)
            pts = pts[mask]
        else:
            pts = points
        return image, pts, crop_size

    def _process_crop(self, image, points_orig):
        """Native-scale clamped RandomResizedCrop -> 224 resize -> density at
        input_size/reduction grid."""
        points = points_orig.copy() if len(points_orig) > 0 else np.empty((0, 2))

        if self.augment:
            image, points, crop_size = self._clamped_random_resized_crop(image.copy(), points)

            if random.random() < 0.5:
                image = TF.hflip(image)
                if len(points) > 0:
                    points[:, 0] = crop_size - 1 - points[:, 0]

            if random.random() < 0.2:
                image = self.color_jitter(image)

            if random.random() < 0.2 and min(image.size) >= 5:
                image = TF.gaussian_blur(image, kernel_size=5, sigma=(0.1, 5.0))

            image = TF.resize(image, (self.input_size, self.input_size))
            if len(points) > 0:
                scale = self.input_size / crop_size
                points[:, 0] *= scale
                points[:, 1] *= scale

            img_tensor = TF.to_tensor(image)

            # Pepper & salt noise (50%, 0.1% of pixels). Mask is per-pixel,
            # broadcast across channels — so a noise pixel flips all 3 channels
            # to white/black, not just one channel to a colored tint.
            if random.random() < 0.5:
                noise = torch.rand(1, *img_tensor.shape[1:])
                img_tensor = torch.where(noise < 0.001, torch.ones_like(img_tensor), img_tensor)
                img_tensor = torch.where(noise > 0.999, torch.zeros_like(img_tensor), img_tensor)
        else:
            # Validation path: plain center resize, no augmentation. Val MAE is
            # computed via sliding-window at native resolution in the training
            # script's evaluate(); this path is kept so DataLoader batching works.
            w, h = image.size
            image = TF.resize(image, (self.input_size, self.input_size))
            if len(points) > 0:
                points[:, 0] *= self.input_size / w
                points[:, 1] *= self.input_size / h
            img_tensor = TF.to_tensor(image)

        img_tensor = self.normalize(img_tensor)

        # Filter points to the valid [0, input_size) x [0, input_size) region
        # BEFORE rasterisation. This guards against float underflow at hflip
        # boundaries (e.g. p[0] = crop_size - eps -> new x = -eps, which would
        # otherwise rasterise at cell 0 via int() truncation toward zero).
        if len(points) > 0:
            valid = ((points[:, 0] >= 0) & (points[:, 0] < self.input_size) &
                     (points[:, 1] >= 0) & (points[:, 1] < self.input_size))
            points = points[valid]

        points_t = torch.tensor(points, dtype=torch.float32) if len(points) > 0 \
                   else torch.zeros((0, 2), dtype=torch.float32)

        density_h = self.input_size // self.reduction
        density_w = self.input_size // self.reduction
        target_density = np.zeros((density_h, density_w), dtype=np.float32)
        if len(points) > 0:
            for p in points:
                x = int(p[0] / self.reduction)
                y = int(p[1] / self.reduction)
                if 0 <= x < density_w and 0 <= y < density_h:
                    target_density[y, x] += 1.0

        # Count-preservation sanity: every filtered point must land in the grid.
        if len(points) > 0:
            assert abs(target_density.sum() - len(points)) < 1e-4, \
                f"density count mismatch: sum={target_density.sum()} vs points={len(points)}"

        target_density = np.clip(target_density, 0, self.max_count)
        target_density = torch.from_numpy(target_density).float().unsqueeze(0)

        return img_tensor, points_t, target_density

    def _load_points(self, mat_path):
        """Load point annotations from a .mat file. Returns (N, 2) array in
        native-pixel coordinates, or (0, 2) on failure."""
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
                    numeric_vals = []
                    for item in raw.flat:
                        if isinstance(item, np.ndarray):
                            numeric_vals.extend(item.flatten().tolist())
                        elif isinstance(item, (int, float)):
                            numeric_vals.append(float(item))
                    if len(numeric_vals) > 0 and len(numeric_vals) % 2 == 0:
                        points = np.array(numeric_vals).reshape(-1, 2).astype(float)
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
                        elif raw.size == 2:
                            points = raw.reshape(1, 2).astype(float)
        except Exception as e:
            print(f"Warning: Failed to load {mat_path}: {e}")
        return points

    def get_raw_sample(self, idx):
        """Return native-resolution image tensor + native points + dataset tag.
        Used by the training script's evaluate() to run sliding-window inference
        at native resolution."""
        img_path, mat_path, tag = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        points = self._load_points(mat_path)
        img_tensor = self.normalize(TF.to_tensor(image))
        return img_tensor, points, tag

    def __getitem__(self, idx):
        img_path, mat_path, _tag = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        points = self._load_points(mat_path)

        crops = [self._process_crop(image, points) for _ in range(self.num_crops)]

        images = torch.stack([c[0] for c in crops], 0)      # [num_crops, C, H, W]
        points_list = [c[1] for c in crops]                  # list of tensors
        densities = torch.stack([c[2] for c in crops], 0)    # [num_crops, 1, H, W]

        return images, points_list, densities


def create_train_val_split(root_dirs, val_fraction=0.1, seed=42, **dataset_kwargs):
    """
    Create train/val datasets with a STRATIFIED split by dataset tag.

    Each dataset (data / drone_data / hacrowd_data / qnrf_converted / ...)
    contributes ceil(n_ds * val_fraction) samples to val. Guards against the
    case where a small dataset (e.g. hacrowd n=13) would otherwise be absent
    from val under a pure-random split, giving noisy per-dataset MAE.

    Returns (train_dataset, val_dataset) as Subset objects.
    Train subset has augmentation ON; val subset has augmentation OFF.
    """
    val_kwargs = {**dataset_kwargs, 'num_crops': 1}
    train_ds = CrowdDataset(root_dirs, augment=True,  **dataset_kwargs)
    val_ds   = CrowdDataset(root_dirs, augment=False, **val_kwargs)

    rng = np.random.default_rng(seed)

    # Group indices by dataset tag, then sample val_fraction from each group.
    by_tag = {}
    for i, (_, _, tag) in enumerate(train_ds.samples):
        by_tag.setdefault(tag, []).append(i)

    val_indices, train_indices = [], []
    for tag, idxs in by_tag.items():
        perm = rng.permutation(len(idxs))
        n_val_ds = max(1, int(round(len(idxs) * val_fraction)))
        shuffled = [idxs[p] for p in perm]
        val_indices.extend(shuffled[:n_val_ds])
        train_indices.extend(shuffled[n_val_ds:])

    # Reshuffle train so batches aren't grouped by dataset tag.
    train_indices = [train_indices[p] for p in rng.permutation(len(train_indices))]

    print(f"Stratified split (seed={seed}):")
    for tag in sorted(by_tag.keys()):
        n_ds = len(by_tag[tag])
        n_val_ds = max(1, int(round(n_ds * val_fraction)))
        print(f"   {tag:<18}  train={n_ds - n_val_ds:>5}  val={n_val_ds:>4}")
    print(f"   {'TOTAL':<18}  train={len(train_indices):>5}  val={len(val_indices):>4}")
    return Subset(train_ds, train_indices), Subset(val_ds, val_indices)


def custom_collate(batch):
    """Custom collate function for multi-crop + variable-length points.
    Each item in batch: (images [num_crops, C, H, W], points_list, densities [num_crops, 1, H, W])
    Flattens crops into the batch dimension.
    """
    all_images, all_points, all_densities = [], [], []
    for imgs, pts_list, dens in batch:
        num_crops = imgs.shape[0]
        for c in range(num_crops):
            all_images.append(imgs[c])
            all_points.append(pts_list[c])
            all_densities.append(dens[c])
    return torch.stack(all_images, 0), all_points, torch.stack(all_densities, 0)
