from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:
    from thop import profile as thop_profile
except ImportError:  # pragma: no cover - optional dependency
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:  # pragma: no cover - optional dependency
    FlopCountAnalysis = None


# -----------------------------
# Data handling
# -----------------------------

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
            orig = fname
            if os.path.isabs(fname) and os.path.isfile(fname):
                resolved.append((fname, int(label)))
                continue

            cand = os.path.join(self.root_dir, fname)
            if os.path.isfile(cand):
                resolved.append((cand, int(label)))
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
                    cand2 = os.path.join(self.root_dir, fname + ext)
                    if os.path.isfile(cand2):
                        resolved.append((cand2, int(label)))
                        break
                else:
                    missing.append(orig)
            else:
                missing.append(orig)

        self.records = resolved
        if missing:
            print(f"[WARN] {len(missing)} images listed in labels were not found under '{self.root_dir}'.")
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
        if "image_filename" in cols and "label" in cols:
            image_col = df.columns[cols.index("image_filename")]
            label_col = df.columns[cols.index("label")]
        elif "id_code" in cols and "diagnosis" in cols:
            image_col = df.columns[cols.index("id_code")]
            label_col = df.columns[cols.index("diagnosis")]
        else:
            image_col, label_col = df.columns[:2]
        df = df[[image_col, label_col]].copy()
        df.columns = ["image_filename", "label"]
        df["label"] = df["label"].astype(int)
        return [(str(a), int(b)) for a, b in df.itertuples(index=False, name=None)]

    rows: List[Tuple[str, int]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((parts[0], int(parts[1])))
    return rows


# -----------------------------
# Model helpers pulled from training scripts
# -----------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetB0WithCBAM(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception:
            self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.cbam = CBAM(in_features)
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.features(x)
        x = self.cbam(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var.add(self.eps).sqrt()) + self.beta


class GSAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_dwd = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=6, dilation=3, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv_pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
        nn.init.zeros_(self.conv_pw.weight)
        if self.conv_pw.bias is not None:
            nn.init.zeros_(self.conv_pw.bias)

    def forward(self, x):
        y = self.act1(self.bn1(self.conv_dw1(x)))
        y = self.act2(self.bn2(self.conv_dwd(y)))
        y = self.conv_pw(y)
        return self.gate(y)


class LSAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.gate(self.conv(attn_map))
        return attn_map


class SAA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gsab = GSAB(in_channels)
        self.lsab = LSAB()

    def forward(self, x_hat):
        f_gsab = x_hat * self.gsab(x_hat)
        f_lsab = x_hat * self.lsab(x_hat)
        return f_gsab + f_lsab


class FFN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.pw_conv.weight)
        if self.pw_conv.bias is not None:
            nn.init.zeros_(self.pw_conv.bias)

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        return residual + x



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
    # Two element-wise multiplications (channel + spatial attention)
    _thop_count_elementwise(module, inputs, ops_per_element=2)


class LSTL(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.saa = SAA(in_channels)
        self.norm2 = LayerNorm2d(in_channels)
        self.ffn = FFN(in_channels)
        self.res1 = nn.Parameter(torch.tensor(0.0))
        self.res2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_hat = self.norm1(x)
        x = x + self.res1 * self.saa(x_hat)
        x_hat = self.norm2(x)
        x = x + self.res2 * self.ffn(x_hat)
        return x


def _count_saa(module, inputs, output):
    # Two element-wise multiplies (with GSAB/LSAB outputs) plus one addition
    _thop_count_elementwise(module, inputs, ops_per_element=3)


def _count_lstl(module, inputs, output):
    # Each residual branch: scalar multiply + addition (2 ops). Two branches => 4 ops/element.
    _thop_count_elementwise(module, inputs, ops_per_element=4)


THOP_CUSTOM_OPS = {
    CBAM: _count_cbam,
    SAA: _count_saa,
    LSTL: _count_lstl,
}


class EfficientNetB0WithLSTL(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = False, insertion_channels: int = 112, probe_img_size: int = 224):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception:
            self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.insert_after_index = None
        self.insertion_channels = insertion_channels
        with torch.no_grad():
            probe = torch.zeros(1, 3, probe_img_size, probe_img_size)
            x = probe
            for i, layer in enumerate(self.base.features):
                x = layer(x)
                c, h, w = x.shape[1], x.shape[2], x.shape[3]
                if self.insert_after_index is None and c == insertion_channels and h >= 14 and w >= 14:
                    self.insert_after_index = i
                    break
            if self.insert_after_index is None:
                self.insert_after_index = len(self.base.features) - 1
                self.insertion_channels = x.shape[1]
        self.lstl = LSTL(in_channels=self.insertion_channels)
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.base.features):
            x = layer(x)
            if i == self.insert_after_index:
                x = self.lstl(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        return x


def build_model(model_key: str, num_classes: int, device: torch.device) -> nn.Module:
    if model_key == "baseline":
        weights = None
        try:
            model = efficientnet_b0(weights=weights)
        except Exception:
            model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_key == "cbam":
        model = EfficientNetB0WithCBAM(num_classes, use_pretrained=False)
    elif model_key == "lstl":
        model = EfficientNetB0WithLSTL(num_classes, use_pretrained=False)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return model.to(device)


# -----------------------------
# FLOPs estimation helpers
# -----------------------------

MODEL_FRIENDLY_NAMES = {
    "baseline": "EfficientNet-B0 (Baseline)",
    "cbam": "EfficientNet-B0 + CBAM",
    "lstl": "EfficientNet-B0 + LSTL",
}


def gflops_static(model_key: Optional[str]) -> Optional[float]:
    approx = {"baseline": 0.414, "cbam": 0.414, "lstl": 0.420}
    return approx.get(model_key) if model_key else None


def measure_model_flops_thop(model: nn.Module, device: torch.device, img_size: int = 224) -> Optional[float]:
    if thop_profile is None:
        return None
    try:
        model_cpu = copy.deepcopy(model).to("cpu")
        dummy = torch.zeros(1, 3, img_size, img_size)
        model_cpu.eval()
        with torch.no_grad():
            macs, _ = thop_profile(model_cpu, inputs=(dummy,), custom_ops=THOP_CUSTOM_OPS, verbose=False)
        gflops_val = float(macs) / 1e9
    except Exception as exc:
        print(f"[WARN] THOP FLOP measurement failed: {exc}")
        gflops_val = None
    finally:
        del model_cpu
    return gflops_val


def measure_model_flops_fvcore(model: nn.Module, device: torch.device, img_size: int = 224) -> Optional[float]:
    if FlopCountAnalysis is None:
        return None
    try:
        model_cpu = copy.deepcopy(model).to("cpu")
        dummy = torch.zeros(1, 3, img_size, img_size)
        with torch.no_grad():
            flops = FlopCountAnalysis(model_cpu, dummy).total()
        gflops_val = float(flops) / 1e9
    except Exception as exc:
        print(f"[WARN] fvcore FLOP measurement failed: {exc}")
        gflops_val = None
    finally:
        del model_cpu
    return gflops_val


def estimate_model_flops(model_key: str, model: nn.Module, device: torch.device) -> Optional[float]:
    measured = measure_model_flops_thop(model, device)
    if measured is not None:
        return measured
    measured = measure_model_flops_fvcore(model, device)
    if measured is not None:
        return measured
    return gflops_static(model_key)


# -----------------------------
# Visualization helpers (unchanged)
# -----------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[int], title: str, out_path: str, normalize: bool = False) -> None:
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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = text_matrix[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center", color="white" if val > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metrics_bar(metrics: Dict[str, float], title: str, out_path: str) -> None:
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color="#2a9d8f")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.03, 1.03), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Evaluation logic
# -----------------------------

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

    ckpt_pat = get_checkpoint_pattern(model_key)
    results: List[EvalResult] = []
    num_classes = len(label_values)

    for fold in range(1, 6):
        model_path = os.path.join(models_dir, ckpt_pat.format(fold))
        if not os.path.isfile(model_path):
            print(f"[WARN] Missing Torch checkpoint for {model_key} fold {fold}: {model_path}")
            continue

        print(f"\n>>> Evaluating {MODEL_FRIENDLY_NAMES[model_key]} - fold {fold} (.pth)")
        model = build_model(model_key, num_classes=num_classes, device=device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as exc:
            print(f"[ERROR] Failed to load weights from {model_path}: {exc}")
            continue
        model.eval()

        gflops_val = estimate_model_flops(model_key, model, device)

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

        fold_dir = ensure_dir(os.path.join(out_dir, f"fold_{fold}"))
        cm_path = os.path.join(fold_dir, "confusion_matrix_counts.csv")
        pd.DataFrame(cm, index=label_values, columns=label_values).to_csv(cm_path)

        severity_columns = [f"Predicted DR Severity - {lbl}" for lbl in label_values]
        per_sample_rows = []
        for idx in range(fold_samples):
            row = {
                "Test": idx + 1,
                "File Name": file_names[idx] if idx < len(file_names) else f"sample_{idx+1}",
                "Actual Severity": int(labels_arr[idx]),
            }
            pred_label = preds_arr[idx]
            for col_name, sev_val in zip(severity_columns, label_values):
                row[col_name] = 1 if pred_label == sev_val else 0
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
        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics_dict, "samples": fold_samples}, f, indent=2)

        res = EvalResult(
            fold=fold,
            samples=fold_samples,
            accuracy=float(acc),
            precision_macro=float(prec),
            recall_macro=float(rec),
            f1_macro=float(f1),
            gflops=gflops_val,
        )
        results.append(res)

    if results:
        df = pd.DataFrame([r.to_row() for r in results])
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


# -----------------------------
# CLI helpers
# -----------------------------

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


def prompt_path(msg: str, validator) -> str:
    while True:
        p = input(msg).strip().strip('\"')
        if not p:
            print("Path cannot be empty. Try again.\n")
            continue
        ap = os.path.abspath(p)
        if validator(ap):
            return ap
        print("Invalid path. Please try again.\n")


def prompt_int_with_default(msg: str, default: int, min_value: int) -> int:
    while True:
        raw = input(msg).strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val < min_value:
                raise ValueError
            return val
        except ValueError:
            print(f"Please enter an integer >= {min_value}.\n")


def infer_default_models_dir(script_path: str) -> str:
    base = os.path.dirname(os.path.abspath(script_path))
    candidates = [
        os.path.normpath(os.path.join(base, "output 2")),
        os.path.normpath(os.path.join(base, "Outputs")),
        os.path.normpath(os.path.join(base, "..", "Main", "Model Files", "pth")),
        os.path.normpath(os.path.join(base, "..", "..", "Main", "Model Files", "pth")),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return base


# -----------------------------
# Entrypoint
# -----------------------------

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()} (device: {device})")

    model_keys = prompt_select_model()

    default_images_dir = r"C:\\Users\\Luigi\\Desktop\\code\\Thesis\\Main\\APTOS 2019"
    default_labels_path = r"C:\\Users\\Luigi\\Desktop\\code\\Thesis\\test.csv"

    while True:
        images_input = input(
            f"Enter path to test images folder (press Enter for default: {default_images_dir}): "
        ).strip().strip('"')
        images_dir = os.path.abspath(images_input) if images_input else default_images_dir
        if os.path.isdir(images_dir):
            break
        print("Invalid images folder. Please try again.\n")

    while True:
        labels_input = input(
            f"Enter path to labels CSV/DAT file (press Enter for default: {default_labels_path}): "
        ).strip().strip('"')
        labels_path = os.path.abspath(labels_input) if labels_input else default_labels_path
        if os.path.isfile(labels_path):
            break
        print("Invalid labels file. Please try again.\n")

    default_models_dir = infer_default_models_dir(__file__)
    print(f"\nDefault Torch checkpoints directory detected: {default_models_dir}")
    models_dir_raw = input("Enter path to .pth models folder (press Enter to use default): ").strip().strip('\"')
    models_dir = os.path.abspath(models_dir_raw) if models_dir_raw else default_models_dir
    if not os.path.isdir(models_dir):
        print(f"[ERROR] Models folder not found: {models_dir}")
        sys.exit(1)

    default_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation_Results")
    print(f"\nDefault output directory: {default_out_dir}")
    out_dir_raw = input("Enter directory for evaluation outputs (press Enter to use default): ").strip().strip('\"')
    out_dir_root = os.path.abspath(out_dir_raw) if out_dir_raw else default_out_dir
    ensure_dir(out_dir_root)

    batch_size = prompt_int_with_default("Enter evaluation batch size [32]: ", default=32, min_value=1)
    num_workers = prompt_int_with_default("Enter DataLoader worker count [2]: ", default=2, min_value=0)

    print("\nLoading labels...")
    labels = read_labels_file(labels_path)
    if not labels:
        print("[ERROR] No labels loaded.")
        sys.exit(1)
    label_values = sorted({int(lbl) for _, lbl in labels})
    print(f"Loaded {len(labels)} samples across labels: {label_values}")

    all_results: Dict[str, List[EvalResult]] = {}
    for key in model_keys:
        model_out_dir = ensure_dir(os.path.join(out_dir_root, key))
        res = evaluate_model_folds(
            model_key=key,
            models_dir=models_dir,
            images_dir=images_dir,
            labels=labels,
            out_dir=model_out_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            label_values=label_values,
            device=device,
        )
        all_results[key] = res

    summary = {k: [r.to_row() for r in v] for k, v in all_results.items() if v}
    with open(os.path.join(out_dir_root, "combined_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results saved under:")
    print(f"  {out_dir_root}")


if __name__ == "__main__":
    main()
