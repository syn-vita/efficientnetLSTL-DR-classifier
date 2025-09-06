# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
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


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(torch.nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetB0WithCBAM(torch.nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}). Falling back to random init.")
            self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.cbam = CBAM(in_features)
        self.base.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.features(x)
        x = self.cbam(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        return x


def main():
    PROJECT_FOLDER = "C:/Users/Luigi/Desktop/code/Thesis/Main"
    IMAGE_FOLDER_NAME = "images"
    CSV_FILE_NAME = "dataset.csv"
    PLOTS_FOLDER = os.path.join(PROJECT_FOLDER, "Figure Outputs", "EfficientNet-B0-CBAM-clean")
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    IMAGE_DIR = os.path.join(PROJECT_FOLDER, IMAGE_FOLDER_NAME)
    ANNOTATIONS_CSV = os.path.join(PROJECT_FOLDER, CSV_FILE_NAME)
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found at: {IMAGE_DIR}")
    if not os.path.isfile(ANNOTATIONS_CSV):
        raise FileNotFoundError(f"Annotations CSV not found at: {ANNOTATIONS_CSV}")

    all_labels_df = pd.read_csv(ANNOTATIONS_CSV, dtype={'id_code': str})
    all_labels_df = all_labels_df.rename(columns={'id_code': 'image_filename', 'diagnosis': 'label'})
    all_labels_df['image_filename'] = all_labels_df['image_filename'] + '.png'
    print(f"Loaded {len(all_labels_df)} records from {ANNOTATIONS_CSV}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_EPOCHS = 60
    NUM_CLASSES = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    N_SPLITS = 5
    NUM_WORKERS = max(2, (os.cpu_count() or 2) - 1)

    USE_PRETRAINED = True
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    BOOST_WEIGHTS = {2: 1.8}

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

        model = EfficientNetB0WithCBAM(NUM_CLASSES, use_pretrained=USE_PRETRAINED).to(device)
        class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)
        if USE_FOCAL_LOSS:
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        trained_model, history, fold_metrics = train_model(
            model, criterion, optimizer, dataloaders, dataset_sizes, device,
            current_fold=fold_num, num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, scheduler=scheduler, patience=8
        )
        all_folds_metrics.extend(fold_metrics)

        model_save_path = os.path.join(PROJECT_FOLDER, f"efficientnet_b0_cbam_clean_fold_{fold_num}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Best model for fold {fold_num} saved to: {model_save_path}")
        plot_save_path = os.path.join(PLOTS_FOLDER, f"Fold{fold_num}_Figure.png")
        plot_history(history, fold_num, plot_save_path)
        print(f"Training history plot for fold {fold_num} saved to: {plot_save_path}")

    print("\n🎉 All folds trained successfully!")
    metrics_df = pd.DataFrame(all_folds_metrics)
    metrics_save_path = os.path.join(PLOTS_FOLDER, "all_folds_detailed_metrics_cbam_clean.csv")
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\n✅ Detailed metrics for all folds saved to: {metrics_save_path}")


if __name__ == '__main__':
    main()
