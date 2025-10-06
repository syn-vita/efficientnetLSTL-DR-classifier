from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

import onnxruntime as ort

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


"""
ONNX inference helpers
"""

def create_onnx_session(model_path: str) -> Tuple[ort.InferenceSession, str, str]:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # add CUDA EP if available
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


# -----------------------------
# Data handling
# -----------------------------

class ImageDataset(Dataset):
    def __init__(self, root_dir: str, records: List[Tuple[str, int]], transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Build an index of all files under root_dir (recursive) for robust lookup
        self._file_index_exact: Dict[str, str] = {}
        self._file_index_lower: Dict[str, str] = {}
        self._file_index_stem: Dict[str, str] = {}
        for dirpath, _, files in os.walk(self.root_dir):
            for fname in files:
                path = os.path.join(dirpath, fname)
                # Exact filename
                self._file_index_exact.setdefault(fname, path)
                # Lowercased filename
                self._file_index_lower.setdefault(fname.lower(), path)
                # Stem (name without extension)
                stem = os.path.splitext(fname)[0]
                self._file_index_stem.setdefault(stem, path)

        # Resolve provided records to existing file paths; skip missing
        resolved: List[Tuple[str, int]] = []
        missing: List[str] = []
        common_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"]
        for fname, label in records:
            orig = fname
            # Absolute path provided
            if os.path.isabs(fname) and os.path.isfile(fname):
                resolved.append((fname, int(label)))
                continue

            # Try join as-is
            cand = os.path.join(self.root_dir, fname)
            if os.path.isfile(cand):
                resolved.append((cand, int(label)))
                continue

            # Try index lookups
            # 1) exact filename
            p = self._file_index_exact.get(fname)
            if p is None:
                # 2) lowercase match
                p = self._file_index_lower.get(fname.lower())
            if p is None:
                # 3) stem match (if no ext provided)
                stem = os.path.splitext(fname)[0]
                p = self._file_index_stem.get(stem)
            if p is not None and os.path.isfile(p):
                resolved.append((p, int(label)))
                continue

            # 4) Try appending common extensions if none supplied
            if os.path.splitext(fname)[1] == "":
                found = False
                for ext in common_exts:
                    cand2 = os.path.join(self.root_dir, fname + ext)
                    if os.path.isfile(cand2):
                        resolved.append((cand2, int(label)))
                        found = True
                        break
                if found:
                    continue

            missing.append(orig)

        self.records = resolved
        if missing:
            print(f"[WARN] {len(missing)} images listed in labels were not found under '{self.root_dir}'.")
            # Print a small sample to help debugging
            for nm in missing[:10]:
                print(f"       - missing: {nm}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label = self.records[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def default_eval_transform(img_size: int = 224):
    try:
        # Prefer torchvision weights transforms if available
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        return weights.transforms(antialias=True)
    except Exception:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def read_labels_file(labels_path: str) -> List[Tuple[str, int]]:
    ext = os.path.splitext(labels_path)[1].lower()
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(labels_path) if ext == ".csv" else pd.read_csv(labels_path, sep="\t")
        cols = [c.lower() for c in df.columns]
        # Normalize columns
        if "image_filename" in cols and "label" in cols:
            image_col = df.columns[cols.index("image_filename")]
            label_col = df.columns[cols.index("label")]
        elif "id_code" in cols and "diagnosis" in cols:
            image_col = df.columns[cols.index("id_code")]
            label_col = df.columns[cols.index("diagnosis")]
        else:
            # try first two columns
            image_col, label_col = df.columns[:2]
        df = df[[image_col, label_col]].copy()
        df.columns = ["image_filename", "label"]
        # Ensure ints
        df["label"] = df["label"].astype(int)
        records = list(df.itertuples(index=False, name=None))
        return [(str(a), int(b)) for a, b in records]
    else:
        # DAT or space-delimited
        rows: List[Tuple[str, int]] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                label = int(parts[1])
                rows.append((fname, label))
        return rows


# -----------------------------
# FLOPs estimation (static fallback)
# -----------------------------

def gflops_static(model_key: str) -> Optional[float]:
    # Approximations based on prior profiling/notes
    approx = {"baseline": 0.414, "cbam": 0.414, "lstl": 0.420}
    return approx.get(model_key)


# -----------------------------
# Evaluation logic
# -----------------------------

@dataclass
class EvalResult:
    fold: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    gflops: Optional[float]


def get_checkpoint_pattern(model_key: str) -> str:
    if model_key == "baseline":
        return "efficientnet_b0_clean_fold_{}.onnx"
    elif model_key == "cbam":
        return "efficientnet_b0_cbam_clean_fold_{}.onnx"
    elif model_key == "lstl":
        return "efficientnet_b0_lstl_clean_fold_{}.onnx"
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def evaluate_model_folds(model_key: str, models_dir: str, images_dir: str, labels: List[Tuple[str, int]], out_dir: str,
                         device: torch.device) -> List[EvalResult]:
    os.makedirs(out_dir, exist_ok=True)
    transform = default_eval_transform(224)

    # Create dataset/dataloader once since test set is constant
    dataset = ImageDataset(images_dir, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=max(2, (os.cpu_count() or 2) - 1), pin_memory=True)

    ckpt_pat = get_checkpoint_pattern(model_key)
    results: List[EvalResult] = []

    # FLOPs (static approximation)
    gflops_val = gflops_static(model_key)

    for fold in range(1, 6):
        model_path = os.path.join(models_dir, ckpt_pat.format(fold))
        if not os.path.isfile(model_path):
            print(f"[WARN] Missing ONNX model for {model_key} fold {fold}: {model_path}")
            continue

        print(f"\n>>> Evaluating {model_key} fold {fold} (ONNX) ...")
        session, input_name, output_name = create_onnx_session(model_path)

        all_preds: List[int] = []
        all_labels: List[int] = []

        # Run images on CPU through ORT (process per image for compatibility with fixed batch dims)
        for images, targets in loader:
            # images: torch.Tensor [B,3,H,W] normalized
            # iterate per image to avoid batch mismatch with ONNX fixed [1,3,H,W]
            for i in range(images.shape[0]):
                img = images[i].numpy()  # float32, [3,H,W]
                img = np.expand_dims(img, axis=0)  # [1,3,H,W]
                outputs = session.run([output_name], {input_name: img})
                logits = outputs[0]  # [1,5]
                pred = int(np.argmax(logits, axis=1)[0])
                all_preds.append(pred)
            all_labels.extend(targets.numpy().astype(int).tolist())

        labels_arr = np.array(all_labels, dtype=int)
        preds_arr = np.array(all_preds, dtype=int)

        cm = confusion_matrix(labels_arr, preds_arr, labels=[0,1,2,3,4])
        acc = accuracy_score(labels_arr, preds_arr)
        prec, rec, f1, _ = precision_recall_fscore_support(labels_arr, preds_arr, labels=[0,1,2,3,4], average='macro', zero_division=0)

        # Save confusion matrix
        cm_path = os.path.join(out_dir, f"confusion_matrix_{model_key}_fold{fold}.csv")
        pd.DataFrame(cm, index=[0,1,2,3,4], columns=[0,1,2,3,4]).to_csv(cm_path)

        # Save metrics row per fold
        res = EvalResult(
            fold=fold,
            accuracy=float(acc),
            precision_macro=float(prec),
            recall_macro=float(rec),
            f1_macro=float(f1),
            gflops=gflops_val,
        )
        results.append(res)

    # Save summary CSV
    if results:
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(os.path.join(out_dir, f"metrics_summary_{model_key}.csv"), index=False)
    return results


def prompt_select_model() -> List[str]:
    print("Select model to evaluate:")
    print("  1) EfficientNet-B0 Baseline")
    print("  2) EfficientNet-B0 + CBAM")
    print("  3) EfficientNet-B0 + LSTL")
    print("  4) All three")
    choice = input("Enter choice [1-4]: ").strip()
    mapping = {
        "1": ["baseline"],
        "2": ["cbam"],
        "3": ["lstl"],
        "4": ["baseline", "cbam", "lstl"],
    }
    return mapping.get(choice, ["baseline"])  # default baseline


def infer_default_models_dir(script_path: str) -> str:
    # Try common locations for exported ONNX files
    base = os.path.dirname(os.path.abspath(script_path))
    candidates = [
        os.path.normpath(os.path.join(base, "Outputs")),
        os.path.normpath(os.path.join(base, "..", "Main", "Outputs")),
        os.path.normpath(os.path.join(base, "..", "dr-classification-webapp", "public", "models")),
        os.path.normpath(os.path.join(base, "..", "..", "dr-classification-webapp", "public", "models")),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return base


def main():
    # Device is irrelevant for ORT CPU; print CUDA availability for info only
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # Model selection
    model_keys = prompt_select_model()

    # Paths
    images_dir = input("Enter path to test images folder: ").strip('"').strip()
    labels_path = input("Enter path to labels CSV/DAT file: ").strip('"').strip()

    # Validate inputs
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images folder not found: {images_dir}")
        sys.exit(1)
    if not os.path.isfile(labels_path):
        print(f"[ERROR] Labels file not found: {labels_path}")
        sys.exit(1)

    # Models directory (ONNX)
    default_models_dir = infer_default_models_dir(__file__)
    print(f"\nDefault ONNX models directory detected: {default_models_dir}")
    models_dir = input(f"Enter path to ONNX models folder [.onnx] (Press Enter to use default): ").strip('"').strip()
    if models_dir == "":
        models_dir = default_models_dir
    if not os.path.isdir(models_dir):
        print(f"[ERROR] Models folder not found: {models_dir}")
        sys.exit(1)

    # Output directory
    out_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation_Results")
    os.makedirs(out_dir_root, exist_ok=True)

    print("\nLoading labels...")
    labels = read_labels_file(labels_path)
    if not labels:
        print("[ERROR] No labels loaded.")
        sys.exit(1)
    # Sanity: ensure labels are in {0..4}
    uniq = sorted(set(int(l) for _, l in labels))
    print(f"Found labels: {uniq}")

    all_results: Dict[str, List[EvalResult]] = {}
    for key in model_keys:
        model_out_dir = os.path.join(out_dir_root, key)
        res = evaluate_model_folds(key, models_dir, images_dir, labels, model_out_dir, device)
        all_results[key] = res

    # Save a combined summary JSON for convenience
    summary = {
        k: [r.__dict__ for r in v]
        for k, v in all_results.items()
    }
    with open(os.path.join(out_dir_root, "combined_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results saved under:")
    print(f"  {out_dir_root}")


if __name__ == "__main__":
    main()
