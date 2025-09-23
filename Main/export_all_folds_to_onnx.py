"""
Export all folds (1..5) of EfficientNet-B0 variants (baseline, CBAM, LSTL) from .pth to .onnx.

Usage examples (PowerShell):
  # Export all variants using defaults (looks in Main/Outputs for .pth, writes .onnx there)
  python Main/export_all_folds_to_onnx.py

  # Export only baseline and cbam to a custom directory
  python Main/export_all_folds_to_onnx.py --models baseline cbam --out-dir C:\\path\\to\\onnx

Assumptions:
- Checkpoints are named:
    baseline: efficientnet_b0_clean_fold_{1..5}.pth
    cbam:     efficientnet_b0_cbam_clean_fold_{1..5}.pth
    lstl:     efficientnet_b0_lstl_clean_fold_{1..5}.pth
- Model definitions for CBAM and LSTL live under "Main/Training Files" as:
    train_efficientnet_b0_cbam.py -> EfficientNetB0WithCBAM
    train_efficientnet_b0_lstl.py -> EfficientNetB0WithLSTL

Outputs:
- Writes .onnx next to or under the chosen output directory with names:
    efficientnet_b0_clean_fold_{i}.onnx
    efficientnet_b0_cbam_clean_fold_{i}.onnx
    efficientnet_b0_lstl_clean_fold_{i}.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

import torch
from torch import nn
from torchvision.models import efficientnet_b0

NUM_CLASSES = 5
IMG_SIZE = 224
FOLDS = [1, 2, 3, 4, 5]


def repo_paths(script_path: str):
    main_dir = os.path.dirname(os.path.abspath(script_path))
    training_dir = os.path.join(main_dir, "Training Files")
    outputs_dir = os.path.join(main_dir, "Outputs")
    return main_dir, training_dir, outputs_dir


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_baseline_fold(pth_path: str, onnx_path: str):
    # Build architecture without downloading pretrained weights
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def export_cbam_fold(training_dir: str, pth_path: str, onnx_path: str):
    if training_dir not in sys.path:
        sys.path.append(training_dir)
    from train_efficientnet_b0_cbam import EfficientNetB0WithCBAM  # type: ignore

    model = EfficientNetB0WithCBAM(num_classes=NUM_CLASSES, use_pretrained=False)
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def export_lstl_fold(training_dir: str, pth_path: str, onnx_path: str):
    if training_dir not in sys.path:
        sys.path.append(training_dir)
    from train_efficientnet_b0_lstl import EfficientNetB0WithLSTL  # type: ignore

    model = EfficientNetB0WithLSTL(num_classes=NUM_CLASSES, use_pretrained=False)
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def do_exports(models: Iterable[str], pth_dir: str, out_dir: str, training_dir: str) -> List[str]:
    ensure_dir(out_dir)
    logs: List[str] = []

    for model_key in models:
        for fold in FOLDS:
            if model_key == "baseline":
                pth = os.path.join(pth_dir, f"efficientnet_b0_clean_fold_{fold}.pth")
                onnx = os.path.join(out_dir, f"efficientnet_b0_clean_fold_{fold}.onnx")
                label = f"baseline fold {fold}"
                exporter = lambda: export_baseline_fold(pth, onnx)
            elif model_key == "cbam":
                pth = os.path.join(pth_dir, f"efficientnet_b0_cbam_clean_fold_{fold}.pth")
                onnx = os.path.join(out_dir, f"efficientnet_b0_cbam_clean_fold_{fold}.onnx")
                label = f"cbam fold {fold}"
                exporter = lambda: export_cbam_fold(training_dir, pth, onnx)
            elif model_key == "lstl":
                pth = os.path.join(pth_dir, f"efficientnet_b0_lstl_clean_fold_{fold}.pth")
                onnx = os.path.join(out_dir, f"efficientnet_b0_lstl_clean_fold_{fold}.onnx")
                label = f"lstl fold {fold}"
                exporter = lambda: export_lstl_fold(training_dir, pth, onnx)
            else:
                continue

            if not os.path.isfile(pth):
                logs.append(f"[SKIP] Missing checkpoint for {label}: {pth}")
                continue
            if os.path.isfile(onnx):
                logs.append(f"[SKIP] ONNX already exists for {label}: {onnx}")
                continue

            try:
                logs.append(f"[EXPORT] {label} -> {onnx}")
                exporter()
                logs.append(f"[OK] {label}")
            except Exception as e:
                logs.append(f"[FAIL] {label}: {e}")
    return logs


def parse_args():
    main_dir, training_dir, outputs_dir = repo_paths(__file__)

    p = argparse.ArgumentParser(description="Export all folds from .pth to .onnx")
    p.add_argument("--models", nargs="+", choices=["baseline", "cbam", "lstl"], default=["baseline", "cbam", "lstl"],
                   help="Model variants to export (default: all)")
    p.add_argument("--pth-dir", default=outputs_dir, help="Directory containing .pth checkpoints (default: Main/Outputs)")
    p.add_argument("--out-dir", default=outputs_dir, help="Directory to write .onnx files (default: Main/Outputs)")
    p.add_argument("--print-only", action="store_true", help="Only print actions without performing export")
    args = p.parse_args()
    # Attach discovered paths for convenience
    args.training_dir = training_dir
    return args


def main():
    args = parse_args()

    actions_preview = []
    for m in args.models:
        for f in FOLDS:
            if m == "baseline":
                pth = os.path.join(args.pth_dir, f"efficientnet_b0_clean_fold_{f}.pth")
                onnx = os.path.join(args.out_dir, f"efficientnet_b0_clean_fold_{f}.onnx")
            elif m == "cbam":
                pth = os.path.join(args.pth_dir, f"efficientnet_b0_cbam_clean_fold_{f}.pth")
                onnx = os.path.join(args.out_dir, f"efficientnet_b0_cbam_clean_fold_{f}.onnx")
            else:  # lstl
                pth = os.path.join(args.pth_dir, f"efficientnet_b0_lstl_clean_fold_{f}.pth")
                onnx = os.path.join(args.out_dir, f"efficientnet_b0_lstl_clean_fold_{f}.onnx")
            actions_preview.append((pth, onnx))

    print("Planned exports:")
    for pth, onnx in actions_preview:
        print(f"  {os.path.basename(pth)} -> {os.path.basename(onnx)}")

    if args.print_only:
        print("--print-only specified, exiting without export.")
        return

    logs = do_exports(args.models, args.pth_dir, args.out_dir, args.training_dir)
    print("\nSummary:")
    for line in logs:
        print(line)


if __name__ == "__main__":
    main()
