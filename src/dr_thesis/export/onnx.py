from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch

from ..models.factory import build_model
from ..paths import CHECKPOINTS_DIR, EXPORTS_DIR, REPO_ROOT

NUM_CLASSES = 5
IMG_SIZE = 224
FOLDS = [1, 2, 3, 4, 5]
MODEL_CHOICES = ("baseline", "cbam", "lstl")


def ensure_dir(path: str | Path) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_checkpoint_name(model_key: str, fold: int) -> str:
    if model_key == "baseline":
        return f"efficientnet_b0_clean_fold_{fold}.pth"
    if model_key == "cbam":
        return f"efficientnet_b0_cbam_clean_fold_{fold}.pth"
    if model_key == "lstl":
        return f"efficientnet_b0_lstl_clean_fold_{fold}.pth"
    raise ValueError(f"Unknown model key: {model_key}")


def get_export_name(model_key: str, fold: int) -> str:
    if model_key == "baseline":
        return f"efficientnet_b0_clean_fold_{fold}.onnx"
    if model_key == "cbam":
        return f"efficientnet_b0_cbam_clean_fold_{fold}.onnx"
    if model_key == "lstl":
        return f"efficientnet_b0_lstl_clean_fold_{fold}.onnx"
    raise ValueError(f"Unknown model key: {model_key}")


def export_fold(model_key: str, pth_path: str | Path, onnx_path: str | Path) -> None:
    model = build_model(model_key, num_classes=NUM_CLASSES, device="cpu", use_pretrained=False)
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


def do_exports(models: Iterable[str], pth_dir: str, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    logs: List[str] = []

    for model_key in models:
        for fold in FOLDS:
            checkpoint_path = Path(pth_dir) / get_checkpoint_name(model_key, fold)
            export_path = Path(out_dir) / get_export_name(model_key, fold)
            label = f"{model_key} fold {fold}"

            if not checkpoint_path.is_file():
                logs.append(f"[SKIP] Missing checkpoint for {label}: {checkpoint_path}")
                continue
            if export_path.is_file():
                logs.append(f"[SKIP] ONNX already exists for {label}: {export_path}")
                continue

            try:
                logs.append(f"[EXPORT] {label} -> {export_path}")
                export_fold(model_key, checkpoint_path, export_path)
                logs.append(f"[OK] {label}")
            except Exception as exc:
                logs.append(f"[FAIL] {label}: {exc}")
    return logs


def has_failed_exports(logs: Iterable[str]) -> bool:
    return any(line.startswith("[FAIL]") for line in logs)


def build_export_plan(models: Iterable[str], pth_dir: str, out_dir: str) -> list[tuple[str, str]]:
    planned: list[tuple[str, str]] = []
    for model_key in models:
        for fold in FOLDS:
            checkpoint_path = Path(pth_dir) / get_checkpoint_name(model_key, fold)
            export_path = Path(out_dir) / get_export_name(model_key, fold)
            planned.append((str(checkpoint_path), str(export_path)))
    return planned


def infer_default_pth_dir() -> str:
    return str(CHECKPOINTS_DIR)


def infer_default_out_dir() -> str:
    return str(EXPORTS_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all checkpoint folds from .pth to .onnx")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="Model variants to export. Default: all.",
    )
    parser.add_argument(
        "--pth-dir",
        default=infer_default_pth_dir(),
        help="Directory containing .pth checkpoints. Default: artifacts/checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        default=infer_default_out_dir(),
        help="Directory to write .onnx files. Default: artifacts/exports.",
    )
    parser.add_argument("--print-only", action="store_true", help="Only print actions without performing export.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    planned_exports = build_export_plan(args.models, args.pth_dir, args.out_dir)

    print("Planned exports:")
    for checkpoint_path, export_path in planned_exports:
        print(f"  {Path(checkpoint_path).name} -> {Path(export_path).name}")

    if args.print_only:
        print("--print-only specified, exiting without export.")
        return 0

    logs = do_exports(args.models, args.pth_dir, args.out_dir)
    print("\nSummary:")
    for line in logs:
        print(line)
    return 1 if has_failed_exports(logs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
