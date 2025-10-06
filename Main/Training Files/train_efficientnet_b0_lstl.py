# -*- coding: utf-8 -*-
import os
from typing import Optional
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
import pandas as pd
try:
    from .train_utils import (
        default_transforms,
        compute_class_weights,
        make_dataloaders,
        train_model,
        plot_history,
        FocalLoss,
    )
except ImportError:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from train_utils import (
        default_transforms,
        compute_class_weights,
        make_dataloaders,
        train_model,
        plot_history,
        FocalLoss,
    )


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
        # Apply each attention mechanism separately and add the results
        f_gsab = x_hat * self.gsab(x_hat)
        f_lsab = x_hat * self.lsab(x_hat)
        return f_gsab + f_lsab  # F_gl = F_GSAB + F_LSAB


class FFN(nn.Module):
    def __init__(self, in_channels, expansion=4):  # expansion kept for API compatibility
        super().__init__()
        # Simpler: residual + PW(Act(DW(x)))
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


class LSTL(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.saa = SAA(in_channels)
        self.norm2 = LayerNorm2d(in_channels)
        self.ffn = FFN(in_channels)
        # ReZero-style residual scalars for stability
        self.res1 = nn.Parameter(torch.tensor(0.0))
        self.res2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_hat = self.norm1(x)
        x = x + self.res1 * self.saa(x_hat)
        x_hat = self.norm2(x)
        x = x + self.res2 * self.ffn(x_hat)
        return x


class EfficientNetB0WithLSTL(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = True, insertion_channels: int = 112, probe_img_size: int = 224):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}). Falling back to random init.")
            self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        # Decide where to insert LSTL based on a probe pass (CPU) to find target channels and >=14x14 spatial size
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
                # Fallback to the last feature layer
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


def main(image_dir: str, csv_path: str, out_dir: Optional[str] = None):
    IMAGE_DIR = os.path.abspath(image_dir)
    ANNOTATIONS_CSV = os.path.abspath(csv_path)
    PROJECT_FOLDER = os.path.abspath(out_dir) if out_dir else os.path.dirname(ANNOTATIONS_CSV)
    PLOTS_FOLDER = os.path.join(PROJECT_FOLDER, "Figure Outputs", "EfficientNet-B0-LSTL-clean")
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found at: {IMAGE_DIR}")
    if not os.path.isfile(ANNOTATIONS_CSV):
        raise FileNotFoundError(f"Annotations CSV not found at: {ANNOTATIONS_CSV}")

    print(f"Using image folder: {IMAGE_DIR}")
    print(f"Using annotations CSV: {ANNOTATIONS_CSV}")
    print(f"Outputs will be saved under: {PROJECT_FOLDER}")

    # Read only id_code and diagnosis; ignore extra columns if present
    try:
        base_df = pd.read_csv(ANNOTATIONS_CSV)
        if {'id_code', 'diagnosis'}.issubset(base_df.columns):
            all_labels_df = base_df[['id_code', 'diagnosis']].copy()
        else:
            base_df = pd.read_csv(ANNOTATIONS_CSV, usecols=[0, 1])
            base_df.columns = ['id_code', 'diagnosis']
            all_labels_df = base_df
    except Exception:
        all_labels_df = pd.read_csv(ANNOTATIONS_CSV, usecols=['id_code', 'diagnosis'])

    all_labels_df['id_code'] = all_labels_df['id_code'].astype(str)
    all_labels_df = all_labels_df.rename(columns={'id_code': 'image_filename', 'diagnosis': 'label'})
    all_labels_df['image_filename'] = all_labels_df['image_filename'] + '.png'
    print(f"Loaded {len(all_labels_df)} records from {ANNOTATIONS_CSV}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_EPOCHS = 60
    NUM_CLASSES = 5
    # Differential learning rates: smaller for backbone, larger for new blocks
    LR_BACKBONE = 3e-5
    LR_HEAD = 1e-4
    WEIGHT_DECAY = 1e-4
    N_SPLITS = 5
    NUM_WORKERS = max(2, (os.cpu_count() or 2) - 1)

    USE_PRETRAINED = True
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    # Avoid boosted weights to prevent double-compensation with a balanced sampler
    BOOST_WEIGHTS = None

    transforms_map = default_transforms(IMG_SIZE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X = all_labels_df['image_filename']
    y = all_labels_df['label']

    print("\nDataset Overview:")
    print(f"Total samples: {len(all_labels_df)}")
    overall_class_counts = all_labels_df['label'].value_counts().sort_index()
    print(f"Class distribution: {overall_class_counts.to_dict()}")
    class_percentages = (overall_class_counts / len(all_labels_df) * 100).round(2)
    print(f"Class percentages: {class_percentages.to_dict()}")
    print("="*50)

    all_folds_metrics = []
    for fold_num, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print("\n" + "="*50)
        print(f"🚀 STARTING FOLD {fold_num}/{N_SPLITS}")
        print("="*50)

        torch.manual_seed(42)
        import numpy as np
        np.random.seed(42)

        train_df = all_labels_df.iloc[train_index].reset_index(drop=True)
        val_df = all_labels_df.iloc[val_index].reset_index(drop=True)
        class_weights = compute_class_weights(train_df, NUM_CLASSES, boost_weights=BOOST_WEIGHTS)
        print(f"Class counts in training set for Fold {fold_num}: {train_df['label'].value_counts().sort_index().to_dict()}")
        print(f"Class weights: {class_weights}")

        dataloaders, dataset_sizes = make_dataloaders(
            train_df, val_df, IMAGE_DIR, transforms_map, BATCH_SIZE, device, NUM_WORKERS, class_weights
        )

        model = EfficientNetB0WithLSTL(NUM_CLASSES, use_pretrained=USE_PRETRAINED).to(device)
        # Use unweighted focal loss to avoid double weighting with a balanced sampler
        if USE_FOCAL_LOSS:
            criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Differential LRs for backbone vs LSTL + classifier
        backbone_params = list(model.base.features.parameters())
        head_params = list(model.lstl.parameters()) + list(model.base.classifier.parameters())
        optimizer = torch.optim.AdamW([
            { 'params': backbone_params, 'lr': LR_BACKBONE },
            { 'params': head_params, 'lr': LR_HEAD },
        ], weight_decay=WEIGHT_DECAY)

        # Warmup then cosine schedule
        warmup_epochs = 3
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - warmup_epochs), eta_min=1e-6),
            ],
            milestones=[warmup_epochs]
        )

        trained_model, history, fold_metrics = train_model(
            model, criterion, optimizer, dataloaders, dataset_sizes, device,
            current_fold=fold_num, num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, scheduler=scheduler, patience=15
        )
        all_folds_metrics.extend(fold_metrics)

        # Export trained model to ONNX instead of saving a .pth checkpoint
        onnx_save_path = os.path.join(PROJECT_FOLDER, f"efficientnet_b0_lstl_clean_fold_{fold_num}.onnx")
        try:
            export_model = trained_model.to("cpu").eval()
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device="cpu")
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_save_path,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            print(f"Best model for fold {fold_num} exported to ONNX: {onnx_save_path}")
        except Exception as e:
            print(f"[ERROR] ONNX export failed for fold {fold_num}: {e}")
        plot_save_path = os.path.join(PLOTS_FOLDER, f"Fold{fold_num}_Figure.png")
        plot_history(history, fold_num, plot_save_path)
        print(f"Training history plot for fold {fold_num} saved to: {plot_save_path}")

    print("\n🎉 All folds trained successfully!")
    metrics_df = pd.DataFrame(all_folds_metrics)
    metrics_save_path = os.path.join(PLOTS_FOLDER, "all_folds_detailed_metrics_lstl_clean.csv")
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\n✅ Detailed metrics for all folds saved to: {metrics_save_path}")


if __name__ == '__main__':
    print("This training module is not meant to be run directly. Please use 'train_cli.py' to launch training.")
    import sys as _sys
    _sys.exit(2)
