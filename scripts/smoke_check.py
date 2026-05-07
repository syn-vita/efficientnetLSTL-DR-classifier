from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LEGACY_PATHS = [
    ROOT / "Main" / "Training Files" / "train_cli.py",
    ROOT / "Main" / "evaluate_trained_folds.py",
    ROOT / "Main" / "export_all_folds_to_onnx.py",
    ROOT / "dr-classification-webapp" / "src" / "config" / "models.js",
]

TASK1_REQUIRED_PATHS = [
    ROOT / "scripts" / "smoke_check.py",
    ROOT / "data" / ".gitkeep",
    ROOT / "data" / "README.md",
    ROOT / "data" / "dataset.csv",
    ROOT / "data" / "train.csv",
    ROOT / "data" / "valid.csv",
    ROOT / "data" / "test.csv",
    ROOT / "docs" / "references" / "README.md",
    ROOT / "docs" / "references" / "BAM Bottleneck Attention Module.pdf",
    ROOT / "docs" / "references" / "EfficientNet with Hybrid Attention Mechanisms for.pdf",
    ROOT / "docs" / "references" / "LiCT-Net_Lightweight_Convolutional_Transformer_Network_for_Multiclass_Breast_Cancer_Classification-1.pdf",
    ROOT / "docs" / "references" / "TW1-Group-9 Paper.pdf",
    ROOT / "artifacts" / "checkpoints" / ".gitkeep",
    ROOT / "artifacts" / "evaluation" / ".gitkeep",
    ROOT / "artifacts" / "exports" / ".gitkeep",
    ROOT / "artifacts" / "figures" / ".gitkeep",
]

TASK1_MOVED_AWAY_PATHS = [
    ROOT / "dataset.csv",
    ROOT / "train.csv",
    ROOT / "valid.csv",
    ROOT / "test.csv",
    ROOT / "Reference Files",
]


def fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    errors = 0
    for path in LEGACY_PATHS:
        if not path.exists():
            errors += fail(f"Missing required path: {path.relative_to(ROOT)}")

    for path in TASK1_REQUIRED_PATHS:
        if not path.exists():
            errors += fail(f"Missing Task 1 path: {path.relative_to(ROOT)}")

    for path in TASK1_MOVED_AWAY_PATHS:
        if path.exists():
            errors += fail(f"Task 1 path should have been moved or removed: {path.relative_to(ROOT)}")

    models_js = ROOT / "dr-classification-webapp" / "src" / "config" / "models.js"
    if models_js.exists():
        text = models_js.read_text(encoding="utf-8")
        for model_name in [
            "/models/efficientnet_b0_clean_fold_3.onnx",
            "/models/efficientnet_b0_cbam_clean_fold_4.onnx",
            "/models/efficientnet_b0_lstl_clean_fold_4.onnx",
        ]:
            if model_name not in text:
                errors += fail(f"Web app config missing expected model path: {model_name}")

    if errors:
        print(f"[FAIL] smoke_check found {errors} issue(s)")
        return 1

    print("[PASS] smoke_check verified Task 1 paths, moved boundaries, and webapp model references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
