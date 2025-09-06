import os
import sys
import csv
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Paths
PROJECT_ROOT = r"C:\Users\Luigi\Desktop\code"
THESIS_MAIN = os.path.join(PROJECT_ROOT, "Thesis", "Main")
TRAINING_DIR = os.path.join(THESIS_MAIN, "Training Files")
WEBAPP_MODELS = os.path.join(PROJECT_ROOT, "Thesis", "dr-classification-webapp", "public", "models")
os.makedirs(WEBAPP_MODELS, exist_ok=True)

# Baseline best fold (from analysis): fold 3
BASELINE_PTH = os.path.join(THESIS_MAIN, "efficientnet_b0_clean_fold_3.pth")
BASELINE_ONNX = os.path.join(WEBAPP_MODELS, "efficientnet_b0_clean_fold_3.onnx")

# LSTL best fold (from analysis): fold 4
LSTL_PTH = os.path.join(THESIS_MAIN, "efficientnet_b0_lstl_clean_fold_4.pth")
LSTL_ONNX = os.path.join(WEBAPP_MODELS, "efficientnet_b0_lstl_clean_fold_4.onnx")

# CBAM metrics CSV to decide the best fold
CBAM_CSV = os.path.join(THESIS_MAIN, "Figure Outputs", "EfficientNet-B0-CBAM-clean", "all_folds_detailed_metrics_cbam_clean.csv")

NUM_CLASSES = 5
IMG_SIZE = 224


def export_baseline():
    if not os.path.isfile(BASELINE_PTH):
        raise FileNotFoundError(f"Missing checkpoint: {BASELINE_PTH}")
    print(f"Loading baseline checkpoint: {BASELINE_PTH}")
    # Recreate model
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    state = torch.load(BASELINE_PTH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Exporting baseline ONNX to: {BASELINE_ONNX}")
    torch.onnx.export(
        model,
        dummy,
        BASELINE_ONNX,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Baseline ONNX export complete.")


def export_lstl():
    if not os.path.isfile(LSTL_PTH):
        raise FileNotFoundError(f"Missing checkpoint: {LSTL_PTH}")
    print(f"Loading LSTL checkpoint: {LSTL_PTH}")
    # Import the model class from your training script (Training Files)
    if TRAINING_DIR not in sys.path:
        sys.path.append(TRAINING_DIR)
    from train_efficientnet_b0_lstl import EfficientNetB0WithLSTL  # type: ignore

    model = EfficientNetB0WithLSTL(num_classes=NUM_CLASSES, use_pretrained=True)
    state = torch.load(LSTL_PTH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Exporting LSTL ONNX to: {LSTL_ONNX}")
    torch.onnx.export(
        model,
        dummy,
        LSTL_ONNX,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("LSTL ONNX export complete.")


def _best_cbam_fold(csv_path: str) -> int:
    """Find the best CBAM fold by maximum balanced_accuracy (take rows with class==0)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CBAM metrics CSV not found: {csv_path}")
    best = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("class") != "0":
                continue
            try:
                fold = int(row["fold"])
                ba = float(row["balanced_accuracy"])
            except Exception:
                continue
            r = best.get(fold)
            if not r or ba > r["balanced_accuracy"]:
                best[fold] = {"fold": fold, "balanced_accuracy": ba}
    if not best:
        raise RuntimeError("No valid CBAM fold records in metrics CSV")
    return max(best.values(), key=lambda x: x["balanced_accuracy"]) ["fold"]


def export_cbam():
    best_fold = _best_cbam_fold(CBAM_CSV)
    cbam_pth = os.path.join(THESIS_MAIN, f"efficientnet_b0_cbam_clean_fold_{best_fold}.pth")
    cbam_onnx = os.path.join(WEBAPP_MODELS, f"efficientnet_b0_cbam_clean_fold_{best_fold}.onnx")
    if not os.path.isfile(cbam_pth):
        raise FileNotFoundError(f"Missing CBAM checkpoint: {cbam_pth}")
    print(f"Loading CBAM checkpoint: {cbam_pth}")

    if TRAINING_DIR not in sys.path:
        sys.path.append(TRAINING_DIR)
    from train_efficientnet_b0_cbam import EfficientNetB0WithCBAM  # type: ignore

    model = EfficientNetB0WithCBAM(num_classes=NUM_CLASSES, use_pretrained=True)
    state = torch.load(cbam_pth, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Exporting CBAM ONNX to: {cbam_onnx}")
    torch.onnx.export(
        model,
        dummy,
        cbam_onnx,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("CBAM ONNX export complete.")


if __name__ == "__main__":
    export_baseline()
    export_lstl()
    export_cbam()
    print("\nAll exports completed. Files saved in:", WEBAPP_MODELS)
