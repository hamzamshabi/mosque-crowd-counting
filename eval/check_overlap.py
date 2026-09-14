"""
Find test images that also appear (exactly or nearly) in the training data.

Each image gets a 64-bit and a 256-bit difference hash (dHash). A pair is a
likely copy when both distances are small; the 64-bit hash alone produces
false matches on low-texture images, so the 256-bit distance confirms it.

Example:
    python eval/check_overlap.py \
        --test "UCF-QNRF_ECCV18/Test/*.jpg" \
        --train qnrf="UCF-QNRF_ECCV18/Train/*.jpg" \
        --train jhu="jhu_crowd_v2.0/*/images/*.jpg" \
        --train nwpu="NWPU-Crowd/images_part*/*.jpg" \
        --out overlap_check.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image


def dhash(path, size):
    im = Image.open(path)
    im.draft("L", (size * 8, size * 8))  # fast JPEG downscale on decode
    px = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).ravel())


def hash_all(paths):
    h64, h256, kept = [], [], []
    for p in paths:
        try:
            h64.append(dhash(p, 8))
            h256.append(dhash(p, 16))
            kept.append(p)
        except OSError as e:
            print(f"skip {p}: {e}")
    return np.array(h64), np.array(h256), kept


def hamming(a, b):
    return np.unpackbits(a[:, None, :] ^ b[None, :, :], axis=2).sum(axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="glob for test images")
    parser.add_argument("--train", action="append", required=True, metavar="NAME=GLOB",
                        help="training set glob; repeat for each source")
    parser.add_argument("--max64", type=int, default=2, help="max 64-bit distance to report")
    parser.add_argument("--max256", type=int, default=16, help="max 256-bit distance to report")
    parser.add_argument("--out", default="overlap_check.csv")
    args = parser.parse_args()

    t64, t256, test_paths = hash_all(sorted(glob.glob(args.test)))
    print(f"test: {len(test_paths)} images")

    rows = []
    for item in args.train:
        name, _, pattern = item.partition("=")
        r64, r256, train_paths = hash_all(sorted(glob.glob(pattern)))
        print(f"{name}: {len(train_paths)} images")
        d64, d256 = hamming(t64, r64), hamming(t256, r256)
        for i, j in zip(*np.nonzero((d64 <= args.max64) & (d256 <= args.max256))):
            rows.append((os.path.basename(test_paths[i]), name, os.path.basename(train_paths[j]),
                         int(d64[i, j]), int(d256[i, j])))

    rows.sort(key=lambda r: (r[3], r[4]))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_image", "train_set", "train_image", "dhash64_dist", "dhash256_dist"])
        w.writerows(rows)
    print(f"{len(rows)} likely copies -> {args.out}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
