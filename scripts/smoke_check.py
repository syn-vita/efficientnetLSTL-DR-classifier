from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACTIVE_PATHS = [
    ROOT / "scripts" / "train.py",
    ROOT / "scripts" / "evaluate.py",
    ROOT / "scripts" / "export_onnx.py",
    ROOT / "scripts" / "smoke_check.py",
    ROOT / "src" / "dr_thesis" / "__init__.py",
    ROOT / "src" / "dr_thesis" / "paths.py",
    ROOT / "dr-classification-webapp" / "src" / "config" / "models.js",
]

TASK1_REQUIRED_PATHS = [
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

ARCHIVE_PATHS = [
    ROOT / "artifacts" / "archive" / "evalresults2",
    ROOT / "artifacts" / "archive" / "figure-outputs",
]

TASK1_MOVED_AWAY_PATHS = [
    ROOT / "dataset.csv",
    ROOT / "train.csv",
    ROOT / "valid.csv",
    ROOT / "test.csv",
    ROOT / "Reference Files",
]

CLEAN_BREAK_REMOVED_PATHS = [
    ROOT / "legacy",
    ROOT / "Main" / "Training Files",
    ROOT / "Main" / "evaluate_trained_folds.py",
    ROOT / "Main" / "export_all_folds_to_onnx.py",
    ROOT / "Main" / "EvalResults2",
    ROOT / "Main" / "Model Files" / "Figure Outputs",
]

FORBIDDEN_SOURCE_REFERENCES = {
    ROOT / "src" / "dr_thesis" / "evaluation" / "folds.py": [
        'REPO_ROOT / "Main" / "output 2"',
        'REPO_ROOT / "Main" / "Outputs"',
        'REPO_ROOT / "Main" / "Model Files" / "pth"',
    ],
    ROOT / "src" / "dr_thesis" / "export" / "onnx.py": [
        'REPO_ROOT / "Main" / "Outputs"',
    ],
}


def fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    errors = 0
    for path in ACTIVE_PATHS:
        if not path.exists():
            errors += fail(f"Missing required path: {path.relative_to(ROOT)}")

    for path in TASK1_REQUIRED_PATHS:
        if not path.exists():
            errors += fail(f"Missing Task 1 path: {path.relative_to(ROOT)}")

    for path in ARCHIVE_PATHS:
        if not path.exists():
            errors += fail(f"Missing archived historical output path: {path.relative_to(ROOT)}")

    for path in TASK1_MOVED_AWAY_PATHS:
        if path.exists():
            errors += fail(f"Task 1 path should have been moved or removed: {path.relative_to(ROOT)}")

    for path in CLEAN_BREAK_REMOVED_PATHS:
        if path.exists():
            errors += fail(f"Legacy path should have been removed: {path.relative_to(ROOT)}")

    for path, snippets in FORBIDDEN_SOURCE_REFERENCES.items():
        if not path.exists():
            errors += fail(f"Missing source file for regression guard: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet in text:
                errors += fail(f"Forbidden legacy path reference still present in {path.relative_to(ROOT)}: {snippet}")

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

    print("[PASS] smoke_check verified active script entry points, cleaned boundaries, source path guards, and webapp model references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
