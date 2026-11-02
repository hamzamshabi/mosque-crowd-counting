"""
Test trained LoRA model on UCF-QNRF test set.
Outputs per-image results and summary to an Excel file.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as io
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

import argparse
import sys

# CLIP-EBC's `models` package must be importable: put your CLIP-EBC checkout on
# PYTHONPATH (see README). lora_utils lives in ../training.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training'))
from models import get_model
from lora_utils import inject_lora_into_clip_vit


@torch.no_grad()
def sliding_window_count(model, image_tensor, window_size=224, stride=224):
    """Sum predicted density over a native-resolution image via non-overlapping
    sliding windows. Zero-pads the bottom/right to a multiple of stride.
    Matches train_lora_vit-b-16.py::_sliding_window_count (used in training-time
    validation), so test-time counts are calibrated with what the model saw."""
    _, _, h, w = image_tensor.shape
    pad_h = max(0, window_size - h) + ((stride - (max(h, window_size) - window_size) % stride) % stride)
    pad_w = max(0, window_size - w) + ((stride - (max(w, window_size) - window_size) % stride) % stride)
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
    _, _, H, W = image_tensor.shape

    total = 0.0
    for top in range(0, H - window_size + 1, stride):
        for left in range(0, W - window_size + 1, stride):
            window = image_tensor[:, :, top:top + window_size, left:left + window_size]
            out = model(window)
            if isinstance(out, tuple):
                _, out = out
            total += float(out.sum().item())
    return total


def load_model(checkpoint_path, backbone_name, device, use_lora=True,
               bins=None, anchor_points=None):
    """Load a model from checkpoint. Reads bins/anchors from checkpoint if available."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Read bins/anchors from checkpoint if not provided
    if 'model_state_dict' in checkpoint:
        if bins is None:
            bins = checkpoint.get('bins', [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))])
        if anchor_points is None:
            anchor_points = checkpoint.get('anchor_points', [0, 1, 2, 3, 4.21937])
    else:
        # Raw pretrained weights — use defaults
        if bins is None:
            bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))]
        if anchor_points is None:
            anchor_points = [0, 1, 2, 3, 4.21937]

    print(f"  Bins: {bins}")
    print(f"  Anchors: {anchor_points}")

    clip_args = {
        'prompt_type': 'word',
        'num_vpt': 32,
        'deep_vpt': True,
        'vpt_drop': 0.0,
    }

    model = get_model(
        backbone=backbone_name,
        input_size=224,
        reduction=8,
        bins=bins,
        anchor_points=anchor_points,
        **clip_args
    )

    if use_lora:
        model = inject_lora_into_clip_vit(model, rank=16)

    model = model.to(device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded {backbone_name} from {checkpoint_path}")
        mae_val = checkpoint.get('mae', None)
        mae_str = f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else "N/A"
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}, MAE: {mae_str}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded {backbone_name} (original pretrained) from {checkpoint_path}")

    model.eval()
    return model


@torch.no_grad()
def test_on_qnrf(model, qnrf_test_dir, device, window_size=224, stride=224):
    """Run inference on QNRF test set using NATIVE-resolution sliding-window —
    same protocol as training-time validation."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # QNRF: images and _ann.mat in same folder
    files = sorted([f for f in os.listdir(qnrf_test_dir) if f.endswith('.jpg')])
    results = []

    for fname in tqdm(files, desc="Testing QNRF"):
        img_path = os.path.join(qnrf_test_dir, fname)
        image = Image.open(img_path).convert('RGB')

        mat_name = os.path.splitext(fname)[0] + '_ann.mat'
        mat_path = os.path.join(qnrf_test_dir, mat_name)
        gt_count = 0
        if os.path.exists(mat_path):
            try:
                mat = io.loadmat(mat_path)
                if 'annPoints' in mat:
                    pts = np.array(mat['annPoints'])
                    gt_count = len(pts)
            except Exception as e:
                print(f"Warning: Failed to load {mat_path}: {e}")

        # Preprocess at native resolution — NO whole-image resize
        img_tensor = TF.to_tensor(image)
        img_tensor = normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Sliding-window inference
        pred_count = sliding_window_count(model, img_tensor, window_size, stride)

        results.append((fname, gt_count, pred_count))

    return results


def write_excel(results, model_name, output_path):
    """Write results to Excel."""
    wb = Workbook()

    header_font = Font(bold=True, size=12, color="FFFFFF")
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    summary_font = Font(bold=True, size=11)
    summary_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    center = Alignment(horizontal='center', vertical='center')

    def style_header(ws, row, num_cols):
        for col in range(1, num_cols + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = thin_border

    def style_cell(ws, row, col, bold=False, fill=None):
        cell = ws.cell(row=row, column=col)
        cell.font = Font(bold=bold, size=11)
        cell.alignment = center
        cell.border = thin_border
        if fill:
            cell.fill = fill

    # ---- Summary Sheet ----
    ws = wb.active
    ws.title = "Summary"

    ws.merge_cells('A1:F1')
    title_cell = ws['A1']
    title_cell.value = f"UCF-QNRF Test - {model_name}"
    title_cell.font = Font(bold=True, size=16, color="2F5496")
    title_cell.alignment = Alignment(horizontal='center')

    gt = np.array([r[1] for r in results])
    pred = np.array([r[2] for r in results])
    errors = pred - gt
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_err = np.mean(errors)
    median_err = np.median(abs_errors)
    std_err = np.std(abs_errors)
    total_gt = np.sum(gt)
    total_pred = np.sum(pred)
    pct_err = abs(total_pred - total_gt) / max(total_gt, 1) * 100

    row = 3
    stats = [
        ("Metric", "Value"),
        ("MAE", f"{mae:.2f}"),
        ("RMSE", f"{rmse:.2f}"),
        ("Mean Error", f"{mean_err:.2f}"),
        ("Median |Error|", f"{median_err:.2f}"),
        ("Std |Error|", f"{std_err:.2f}"),
        ("Min |Error|", f"{np.min(abs_errors):.2f}"),
        ("Max |Error|", f"{np.max(abs_errors):.2f}"),
        ("Total GT Count", f"{total_gt:.0f}"),
        ("Total Predicted", f"{total_pred:.1f}"),
        ("Total % Error", f"{pct_err:.1f}%"),
        ("Test Images", f"{len(results)}"),
    ]

    for i, (label, val) in enumerate(stats):
        r = row + i
        ws.cell(row=r, column=1, value=label)
        ws.cell(row=r, column=2, value=val)
        if i == 0:
            style_header(ws, r, 2)
        else:
            style_cell(ws, r, 1)
            style_cell(ws, r, 2)

    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 15

    # ---- Per-image Details Sheet ----
    ws2 = wb.create_sheet(title="Details")

    headers = ["#", "Image", "GT Count", "Predicted", "Error", "|Error|", "% Error"]
    for col_idx, h in enumerate(headers, 1):
        ws2.cell(row=1, column=col_idx, value=h)
    style_header(ws2, 1, len(headers))

    for i, (fname, gt_c, pred_c) in enumerate(results):
        row = i + 2
        error = pred_c - gt_c
        abs_error = abs(error)
        pct = (abs_error / max(gt_c, 1)) * 100

        ws2.cell(row=row, column=1, value=i + 1)
        ws2.cell(row=row, column=2, value=fname)
        ws2.cell(row=row, column=3, value=gt_c)
        ws2.cell(row=row, column=4, value=round(pred_c, 2))
        ws2.cell(row=row, column=5, value=round(error, 2))
        ws2.cell(row=row, column=6, value=round(abs_error, 2))
        ws2.cell(row=row, column=7, value=f"{pct:.1f}%")

        for col in range(1, 8):
            style_cell(ws2, row, col)

        # Highlight large errors (> 2x MAE)
        if abs_error > mae * 2:
            err_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            for col in range(1, 8):
                ws2.cell(row=row, column=col).fill = err_fill

    # Summary at bottom
    row = len(results) + 3
    for label, val in [("MAE:", f"{mae:.2f}"), ("RMSE:", f"{rmse:.2f}")]:
        ws2.cell(row=row, column=2, value=label)
        ws2.cell(row=row, column=3, value=val)
        ws2.cell(row=row, column=2).font = summary_font
        ws2.cell(row=row, column=3).font = summary_font
        ws2.cell(row=row, column=2).fill = summary_fill
        ws2.cell(row=row, column=3).fill = summary_fill
        row += 1

    widths = [6, 25, 12, 12, 10, 10, 12]
    for col, w in enumerate(widths, 1):
        ws2.column_dimensions[get_column_letter(col)].width = w

    wb.save(output_path)
    print(f"\nResults saved to: {output_path}")


def write_combined_excel(all_results, output_path):
    """Write comparison results for multiple models to Excel."""
    wb = Workbook()

    header_font = Font(bold=True, size=12, color="FFFFFF")
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    summary_font = Font(bold=True, size=11)
    summary_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    best_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    center = Alignment(horizontal='center', vertical='center')

    def style_header(ws, row, num_cols):
        for col in range(1, num_cols + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = thin_border

    def style_cell(ws, row, col, bold=False, fill=None):
        cell = ws.cell(row=row, column=col)
        cell.font = Font(bold=bold, size=11)
        cell.alignment = center
        cell.border = thin_border
        if fill:
            cell.fill = fill

    # ---- Summary Sheet ----
    ws = wb.active
    ws.title = "Summary"

    ws.merge_cells('A1:G1')
    title_cell = ws['A1']
    title_cell.value = "UCF-QNRF Test - Model Comparison"
    title_cell.font = Font(bold=True, size=16, color="2F5496")
    title_cell.alignment = Alignment(horizontal='center')

    row = 3
    headers = ["Model", "MAE", "RMSE", "Mean Error", "Median |Error|", "Total GT", "Total Pred"]
    for col_idx, h in enumerate(headers, 1):
        ws.cell(row=row, column=col_idx, value=h)
    style_header(ws, row, len(headers))

    best_mae = float('inf')
    best_model = ""
    for name, results in all_results.items():
        gt = np.array([r[1] for r in results])
        pred = np.array([r[2] for r in results])
        mae = np.mean(np.abs(pred - gt))
        if mae < best_mae:
            best_mae = mae
            best_model = name

    for name, results in all_results.items():
        row += 1
        gt = np.array([r[1] for r in results])
        pred = np.array([r[2] for r in results])
        errors = pred - gt
        abs_errors = np.abs(errors)

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mean_err = np.mean(errors)
        median_err = np.median(abs_errors)

        values = [name, f"{mae:.2f}", f"{rmse:.2f}", f"{mean_err:.2f}",
                  f"{median_err:.2f}", f"{np.sum(gt):.0f}", f"{np.sum(pred):.1f}"]
        for col_idx, v in enumerate(values, 1):
            ws.cell(row=row, column=col_idx, value=v)
            fill = best_fill if name == best_model else None
            style_cell(ws, row, col_idx, fill=fill)

    for col in range(1, 8):
        ws.column_dimensions[get_column_letter(col)].width = 22

    # ---- Per-model detail sheets ----
    for name, results in all_results.items():
        # Shorten sheet name (max 31 chars), remove invalid chars
        sheet_name = name.replace('/', '-').replace('\\', '-')[:31]
        ws2 = wb.create_sheet(title=sheet_name)

        headers = ["#", "Image", "GT Count", "Predicted", "Error", "|Error|", "% Error"]
        for col_idx, h in enumerate(headers, 1):
            ws2.cell(row=1, column=col_idx, value=h)
        style_header(ws2, 1, len(headers))

        gt = np.array([r[1] for r in results])
        pred = np.array([r[2] for r in results])
        mae = np.mean(np.abs(pred - gt))

        for i, (fname, gt_c, pred_c) in enumerate(results):
            r = i + 2
            error = pred_c - gt_c
            abs_error = abs(error)
            pct = (abs_error / max(gt_c, 1)) * 100

            ws2.cell(row=r, column=1, value=i + 1)
            ws2.cell(row=r, column=2, value=fname)
            ws2.cell(row=r, column=3, value=gt_c)
            ws2.cell(row=r, column=4, value=round(pred_c, 2))
            ws2.cell(row=r, column=5, value=round(error, 2))
            ws2.cell(row=r, column=6, value=round(abs_error, 2))
            ws2.cell(row=r, column=7, value=f"{pct:.1f}%")

            for col in range(1, 8):
                style_cell(ws2, r, col)

            if abs_error > mae * 2:
                err_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                for col in range(1, 8):
                    ws2.cell(row=r, column=col).fill = err_fill

        r = len(results) + 3
        errors = pred - gt
        abs_errors = np.abs(errors)
        for label, val in [("MAE:", f"{np.mean(abs_errors):.2f}"), ("RMSE:", f"{np.sqrt(np.mean(errors**2)):.2f}")]:
            ws2.cell(row=r, column=2, value=label)
            ws2.cell(row=r, column=3, value=val)
            ws2.cell(row=r, column=2).font = summary_font
            ws2.cell(row=r, column=3).font = summary_font
            ws2.cell(row=r, column=2).fill = summary_fill
            ws2.cell(row=r, column=3).fill = summary_fill
            r += 1

        widths = [6, 25, 12, 12, 10, 10, 12]
        for col, w in enumerate(widths, 1):
            ws2.column_dimensions[get_column_letter(col)].width = w

    wb.save(output_path)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP-EBC checkpoints on the UCF-QNRF test set (native-resolution protocol).")
    parser.add_argument('--qnrf-dir', required=True, help='UCF-QNRF_ECCV18/Test folder (img_XXXX.jpg + _ann.mat)')
    parser.add_argument('--baseline', help='original pretrained CLIP-EBC ViT-B/16 weights')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                        help='fine-tuned LoRA checkpoint; repeat to compare several')
    parser.add_argument('--out', default='qnrf_test_results.xlsx')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qnrf_test_dir = args.qnrf_dir
    output_path = args.out

    all_results = {}

    models_to_test = []
    if args.baseline:
        models_to_test.append(("Original B/16 (pretrained)", args.baseline, False))
    for item in args.checkpoint:
        name, _, path = item.rpartition('=')
        models_to_test.append((name or os.path.basename(path), path, True))

    for name, ckpt_path, use_lora in models_to_test:
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {name} — {ckpt_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Testing {name} on UCF-QNRF")
        print(f"{'=' * 60}")

        model = load_model(ckpt_path, 'clip_vit_b_16', device, use_lora=use_lora)
        results = test_on_qnrf(model, qnrf_test_dir, device)
        del model
        torch.cuda.empty_cache()

        gt = np.array([r[1] for r in results])
        pred = np.array([r[2] for r in results])
        print(f"\n  {name} QNRF MAE:  {np.mean(np.abs(pred - gt)):.2f}")
        print(f"  {name} QNRF RMSE: {np.sqrt(np.mean((pred - gt)**2)):.2f}")
        all_results[name] = results

    # ---- Write combined Excel ----
    write_combined_excel(all_results, output_path)
    print(f"\nDone! Open {output_path} to see results.")


if __name__ == '__main__':
    main()
