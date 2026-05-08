from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

from ..data.records import read_eval_labels
from ..models.cbam import CBAM
from ..models.factory import build_model
from ..models.lstl import LSTL, SAA
from ..paths import CHECKPOINTS_DIR, EVALUATION_DIR, REPO_ROOT

try:
    from thop import profile as thop_profile
except ImportError:  # pragma: no cover - optional dependency
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:  # pragma: no cover - optional dependency
    FlopCountAnalysis = None

MODEL_FRIENDLY_NAMES = {
    "baseline": "EfficientNet-B0 (Baseline)",
    "cbam": "EfficientNet-B0 + CBAM",
    "lstl": "EfficientNet-B0 + LSTL",
}
MODEL_CHOICES = ("baseline", "cbam", "lstl")
DEFAULT_IMAGES_DIR = REPO_ROOT / "Main" / "APTOS 2019"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "test.csv"


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, records: List[Tuple[str, int]], transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self._file_index_exact: Dict[str, str] = {}
        self._file_index_lower: Dict[str, str] = {}
        self._file_index_stem: Dict[str, str] = {}
        for dirpath, _, files in os.walk(self.root_dir):
            for fname in files:
                path = os.path.join(dirpath, fname)
                self._file_index_exact.setdefault(fname, path)
                self._file_index_lower.setdefault(fname.lower(), path)
                stem = os.path.splitext(fname)[0]
                self._file_index_stem.setdefault(stem, path)

        resolved: List[Tuple[str, int]] = []
        missing: List[str] = []
        common_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"]
        for fname, label in records:
            original_name = fname
            if os.path.isabs(fname) and os.path.isfile(fname):
                resolved.append((fname, int(label)))
                continue

            candidate = os.path.join(self.root_dir, fname)
            if os.path.isfile(candidate):
                resolved.append((candidate, int(label)))
                continue

            found = self._file_index_exact.get(fname) or self._file_index_lower.get(fname.lower())
            if found is None:
                stem = os.path.splitext(fname)[0]
                found = self._file_index_stem.get(stem)
            if found and os.path.isfile(found):
                resolved.append((found, int(label)))
                continue

            if os.path.splitext(fname)[1] == "":
                for ext in common_exts:
                    candidate = os.path.join(self.root_dir, fname + ext)
                    if os.path.isfile(candidate):
                        resolved.append((candidate, int(label)))
                        break
                else:
                    missing.append(original_name)
            else:
                missing.append(original_name)

        self.records = resolved
        if missing:
            print(f"[WARN] {len(missing)} images listed in labels were not found under '{self.root_dir}'.")
            for name in missing[:10]:
                print(f"       - missing: {name}")

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
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        return weights.transforms(antialias=True)
    except Exception:
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def _thop_first_tensor(inputs):
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            if torch.is_tensor(item):
                return item
    elif torch.is_tensor(inputs):
        return inputs
    raise ValueError("Expected tensor input for THOP custom op handler")


def _thop_count_elementwise(module, inputs, ops_per_element: int) -> None:
    tensor = _thop_first_tensor(inputs)
    numel = tensor.numel()
    if not hasattr(module, "total_ops"):
        module.total_ops = torch.DoubleTensor([0.0])
    module.total_ops += torch.DoubleTensor([int(numel * ops_per_element)])


def _count_cbam(module, inputs, output):
    del output
    _thop_count_elementwise(module, inputs, ops_per_element=2)


def _count_saa(module, inputs, output):
    del output
    _thop_count_elementwise(module, inputs, ops_per_element=3)


def _count_lstl(module, inputs, output):
    del output
    _thop_count_elementwise(module, inputs, ops_per_element=4)


THOP_CUSTOM_OPS = {
    CBAM: _count_cbam,
    SAA: _count_saa,
    LSTL: _count_lstl,
}


def gflops_static(model_key: Optional[str]) -> Optional[float]:
    approx = {"baseline": 0.414, "cbam": 0.414, "lstl": 0.420}
    return approx.get(model_key) if model_key else None


def measure_model_flops_thop(model: nn.Module, img_size: int = 224) -> Optional[float]:
    if thop_profile is None:
        return None
    model_cpu = None
    try:
        model_cpu = copy.deepcopy(model).to("cpu")
        dummy = torch.zeros(1, 3, img_size, img_size)
        model_cpu.eval()
        with torch.no_grad():
            macs, _ = thop_profile(model_cpu, inputs=(dummy,), custom_ops=THOP_CUSTOM_OPS, verbose=False)
        return float(macs) / 1e9
    except Exception as exc:
        print(f"[WARN] THOP FLOP measurement failed: {exc}")
        return None
    finally:
        del model_cpu


def measure_model_flops_fvcore(model: nn.Module, img_size: int = 224) -> Optional[float]:
    if FlopCountAnalysis is None:
        return None
    model_cpu = None
    try:
        model_cpu = copy.deepcopy(model).to("cpu")
        dummy = torch.zeros(1, 3, img_size, img_size)
        with torch.no_grad():
            flops = FlopCountAnalysis(model_cpu, dummy).total()
        return float(flops) / 1e9
    except Exception as exc:
        print(f"[WARN] fvcore FLOP measurement failed: {exc}")
        return None
    finally:
        del model_cpu


def estimate_model_flops(model_key: str, model: nn.Module) -> Optional[float]:
    measured = measure_model_flops_thop(model)
    if measured is not None:
        return measured
    measured = measure_model_flops_fvcore(model)
    if measured is not None:
        return measured
    return gflops_static(model_key)


def ensure_dir(path: str | Path) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int],
    title: str,
    out_path: str | Path,
    normalize: bool = False,
) -> None:
    data = cm.astype(float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sum = data.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            data = data / row_sum
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fmt = ".2f" if normalize else "d"
    text_matrix = data if normalize else cm
    thresh = data.max() / 2.0 if data.size else 0.0
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = text_matrix[row, col]
            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metrics_bar(metrics: Dict[str, float], title: str, out_path: str | Path) -> None:
    keys = list(metrics.keys())
    values = [metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, values, color="#2a9d8f")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.03, 1.03),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


@dataclass
class EvalResult:
    fold: int
    samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    gflops: Optional[float]

    def to_row(self) -> Dict[str, object]:
        return {
            "fold": self.fold,
            "samples": self.samples,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "gflops": self.gflops,
        }


def get_checkpoint_pattern(model_key: str) -> str:
    patterns = {
        "baseline": "efficientnet_b0_clean_fold_{}.pth",
        "cbam": "efficientnet_b0_cbam_clean_fold_{}.pth",
        "lstl": "efficientnet_b0_lstl_clean_fold_{}.pth",
    }
    if model_key not in patterns:
        raise ValueError(f"Unknown model key: {model_key}")
    return patterns[model_key]


def evaluate_model_folds(
    model_key: str,
    models_dir: str,
    images_dir: str,
    labels: List[Tuple[str, int]],
    out_dir: str,
    batch_size: int,
    num_workers: int,
    label_values: Sequence[int],
    device: torch.device,
) -> List[EvalResult]:
    ensure_dir(out_dir)
    transform = default_eval_transform(224)

    dataset = ImageDataset(images_dir, labels, transform=transform)
    if len(dataset) == 0:
        print(f"[WARN] No images resolved for dataset under {images_dir}. Skipping {model_key}.")
        return []

    file_names = [os.path.basename(path) for path, _ in dataset.records]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_pattern = get_checkpoint_pattern(model_key)
    results: List[EvalResult] = []
    num_classes = len(label_values)

    for fold in range(1, 6):
        model_path = os.path.join(models_dir, checkpoint_pattern.format(fold))
        if not os.path.isfile(model_path):
            print(f"[WARN] Missing Torch checkpoint for {model_key} fold {fold}: {model_path}")
            continue

        print(f"\n>>> Evaluating {MODEL_FRIENDLY_NAMES[model_key]} - fold {fold} (.pth)")
        model = build_model(model_key, num_classes=num_classes, device=device, use_pretrained=False)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as exc:
            print(f"[ERROR] Failed to load weights from {model_path}: {exc}")
            continue
        model.eval()

        gflops_value = estimate_model_flops(model_key, model)
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images_batch, targets in loader:
                images_batch = images_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().astype(int).tolist())
                all_labels.extend(targets.cpu().numpy().astype(int).tolist())

        labels_arr = np.array(all_labels, dtype=int)
        preds_arr = np.array(all_preds, dtype=int)
        fold_samples = int(labels_arr.size)
        if fold_samples == 0:
            print(f"[WARN] No samples evaluated for fold {fold}.")
            continue

        cm = confusion_matrix(labels_arr, preds_arr, labels=list(label_values))
        acc = accuracy_score(labels_arr, preds_arr)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_arr,
            preds_arr,
            labels=list(label_values),
            average="macro",
            zero_division=0,
        )

        fold_dir = ensure_dir(Path(out_dir) / f"fold_{fold}")
        pd.DataFrame(cm, index=label_values, columns=label_values).to_csv(
            os.path.join(fold_dir, "confusion_matrix_counts.csv")
        )

        severity_columns = [f"Predicted DR Severity - {label}" for label in label_values]
        per_sample_rows = []
        for idx in range(fold_samples):
            row = {
                "Test": idx + 1,
                "File Name": file_names[idx] if idx < len(file_names) else f"sample_{idx + 1}",
                "Actual Severity": int(labels_arr[idx]),
            }
            predicted_label = preds_arr[idx]
            for col_name, severity_value in zip(severity_columns, label_values):
                row[col_name] = 1 if predicted_label == severity_value else 0
            per_sample_rows.append(row)
        pd.DataFrame(per_sample_rows).to_csv(os.path.join(fold_dir, "test_results.csv"), index=False)

        plot_confusion_matrix(
            cm,
            labels=label_values,
            title=f"{MODEL_FRIENDLY_NAMES[model_key]} - Fold {fold} (Counts)",
            out_path=os.path.join(fold_dir, "confusion_matrix.png"),
            normalize=False,
        )
        plot_confusion_matrix(
            cm,
            labels=label_values,
            title=f"{MODEL_FRIENDLY_NAMES[model_key]} - Fold {fold} (Normalized)",
            out_path=os.path.join(fold_dir, "confusion_matrix_normalized.png"),
            normalize=True,
        )

        metrics_dict = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        plot_metrics_bar(
            metrics_dict,
            title=f"{MODEL_FRIENDLY_NAMES[model_key]} - Fold {fold} Metrics",
            out_path=os.path.join(fold_dir, "metrics_overview.png"),
        )
        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics_dict, "samples": fold_samples}, handle, indent=2)

        results.append(
            EvalResult(
                fold=fold,
                samples=fold_samples,
                accuracy=float(acc),
                precision_macro=float(prec),
                recall_macro=float(rec),
                f1_macro=float(f1),
                gflops=gflops_value,
            )
        )

    if results:
        df = pd.DataFrame([result.to_row() for result in results])
        mean_row = {
            "fold": "mean",
            "samples": df["samples"].mean(),
            "accuracy": df["accuracy"].mean(),
            "precision_macro": df["precision_macro"].mean(),
            "recall_macro": df["recall_macro"].mean(),
            "f1_macro": df["f1_macro"].mean(),
            "gflops": df["gflops"].dropna().mean() if df["gflops"].notna().any() else None,
        }
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df.to_csv(os.path.join(out_dir, f"metrics_summary_{model_key}.csv"), index=False)

    return results


def prompt_select_model() -> List[str]:
    print("Select model(s) to evaluate:")
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
    return mapping.get(choice, ["baseline"])


def prompt_path(message: str, validator) -> str:
    while True:
        path_value = input(message).strip().strip('"')
        if not path_value:
            print("Path cannot be empty. Try again.\n")
            continue
        absolute_path = os.path.abspath(path_value)
        if validator(absolute_path):
            return absolute_path
        print("Invalid path. Please try again.\n")


def prompt_int_with_default(message: str, default: int, min_value: int) -> int:
    while True:
        raw = input(message).strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if value < min_value:
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter an integer >= {min_value}.\n")


def infer_default_models_dir(script_path: str | None = None) -> str:
    del script_path
    candidates = [
        CHECKPOINTS_DIR,
        REPO_ROOT / "Main" / "output 2",
        REPO_ROOT / "Main" / "Outputs",
        REPO_ROOT / "Main" / "Model Files" / "pth",
        REPO_ROOT / "output 2",
        REPO_ROOT / "Outputs",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(CHECKPOINTS_DIR)


def infer_default_images_dir() -> str:
    candidates = [
        REPO_ROOT / "APTOS 2019",
        DEFAULT_IMAGES_DIR,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(DEFAULT_IMAGES_DIR)


def infer_default_labels_path() -> str:
    candidates = [
        DEFAULT_LABELS_PATH,
        REPO_ROOT / "test.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(DEFAULT_LABELS_PATH)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoint folds for EfficientNet-B0 variants.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        help="Model variants to evaluate. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument("--images-dir", help="Path to the evaluation images directory.")
    parser.add_argument("--labels-path", help="Path to the labels file (.csv, .tsv, or .dat).")
    parser.add_argument(
        "--models-dir",
        help="Path to the directory containing .pth checkpoints. Defaults to the detected repo checkpoints folder.",
    )
    parser.add_argument(
        "--out-dir",
        help="Path to the root evaluation output directory. Defaults to artifacts/evaluation when available.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size. Default: 32.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count. Default: 2.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Default: auto.",
    )
    return parser.parse_args(argv)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_inputs(args: argparse.Namespace) -> tuple[List[str], str, str, str, str, int, int]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    model_keys = list(args.models) if args.models else prompt_select_model()

    default_images_dir = infer_default_images_dir()
    if args.images_dir:
        images_dir = os.path.abspath(args.images_dir)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
    else:
        while True:
            user_input = input(
                f"Enter path to test images folder (press Enter for default: {default_images_dir}): "
            ).strip().strip('"')
            images_dir = os.path.abspath(user_input) if user_input else default_images_dir
            if os.path.isdir(images_dir):
                break
            print("Invalid images folder. Please try again.\n")

    default_labels_path = infer_default_labels_path()
    if args.labels_path:
        labels_path = os.path.abspath(args.labels_path)
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
    else:
        while True:
            user_input = input(
                f"Enter path to labels CSV/DAT file (press Enter for default: {default_labels_path}): "
            ).strip().strip('"')
            labels_path = os.path.abspath(user_input) if user_input else default_labels_path
            if os.path.isfile(labels_path):
                break
            print("Invalid labels file. Please try again.\n")

    default_models_dir = infer_default_models_dir()
    if args.models_dir:
        models_dir = os.path.abspath(args.models_dir)
    else:
        print(f"\nDefault Torch checkpoints directory detected: {default_models_dir}")
        raw_models_dir = input("Enter path to .pth models folder (press Enter to use default): ").strip().strip('"')
        models_dir = os.path.abspath(raw_models_dir) if raw_models_dir else default_models_dir
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models folder not found: {models_dir}")

    default_out_dir = str(EVALUATION_DIR)
    if args.out_dir:
        out_dir_root = os.path.abspath(args.out_dir)
    else:
        print(f"\nDefault output directory: {default_out_dir}")
        raw_out_dir = input("Enter directory for evaluation outputs (press Enter to use default): ").strip().strip('"')
        out_dir_root = os.path.abspath(raw_out_dir) if raw_out_dir else default_out_dir
    ensure_dir(out_dir_root)

    return model_keys, images_dir, labels_path, models_dir, out_dir_root, args.batch_size, args.num_workers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()} (device: {device})")

    model_keys, images_dir, labels_path, models_dir, out_dir_root, batch_size, num_workers = resolve_inputs(args)

    print("\nLoading labels...")
    labels = read_eval_labels(labels_path)
    if not labels:
        print("[ERROR] No labels loaded.")
        return 1
    label_values = sorted({int(label) for _, label in labels})
    print(f"Loaded {len(labels)} samples across labels: {label_values}")

    all_results: Dict[str, List[EvalResult]] = {}
    for model_key in model_keys:
        model_out_dir = ensure_dir(Path(out_dir_root) / model_key)
        results = evaluate_model_folds(
            model_key=model_key,
            models_dir=models_dir,
            images_dir=images_dir,
            labels=labels,
            out_dir=model_out_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            label_values=label_values,
            device=device,
        )
        all_results[model_key] = results

    summary = {key: [result.to_row() for result in value] for key, value in all_results.items() if value}
    with open(os.path.join(out_dir_root, "combined_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nDone. Results saved under:")
    print(f"  {out_dir_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
