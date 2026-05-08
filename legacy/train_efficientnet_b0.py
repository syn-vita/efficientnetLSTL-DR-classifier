# -*- coding: utf-8 -*-
import os
from typing import Optional
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
    # Fallback for direct execution (no package context)
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


def main(image_dir: str, csv_path: str, out_dir: Optional[str] = None):
    IMAGE_DIR = os.path.abspath(image_dir)
    ANNOTATIONS_CSV = os.path.abspath(csv_path)
    PROJECT_FOLDER = os.path.abspath(out_dir) if out_dir else os.path.dirname(ANNOTATIONS_CSV)
    PLOTS_FOLDER = os.path.join(PROJECT_FOLDER, "Figure Outputs", "EfficientNet-B0-clean")
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
            # Fallback to first two columns and rename
            base_df = pd.read_csv(ANNOTATIONS_CSV, usecols=[0, 1])
            base_df.columns = ['id_code', 'diagnosis']
            all_labels_df = base_df
    except Exception:
        # Last resort, try strict selection by name
        all_labels_df = pd.read_csv(ANNOTATIONS_CSV, usecols=['id_code', 'diagnosis'])

    all_labels_df['id_code'] = all_labels_df['id_code'].astype(str)
    all_labels_df = all_labels_df.rename(columns={'id_code': 'image_filename', 'diagnosis': 'label'})
    all_labels_df['image_filename'] = all_labels_df['image_filename'] + '.png'
    print(f"Loaded {len(all_labels_df)} records from {ANNOTATIONS_CSV}")

    # Params
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

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
        try:
            model = efficientnet_b0(weights=weights).to(device)
        except Exception as e:
            print(f"Warning: failed to load pretrained weights ({e}). Falling back to random init.")
            model = efficientnet_b0(weights=None).to(device)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES).to(device)

        class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)
        if USE_FOCAL_LOSS:
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        trained_model, history, fold_metrics = train_model(
            model, criterion, optimizer, dataloaders, dataset_sizes, device,
            current_fold=fold_num, num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, scheduler=scheduler, patience=7
        )
        all_folds_metrics.extend(fold_metrics)

        # Persist checkpoints in both TorchScript (.pth) and ONNX formats
        pth_save_path = os.path.join(PROJECT_FOLDER, f"efficientnet_b0_clean_fold_{fold_num}.pth")
        try:
            torch.save(trained_model.state_dict(), pth_save_path)
            print(f"Best model for fold {fold_num} saved to: {pth_save_path}")
        except Exception as e:
            print(f"[ERROR] Torch checkpoint save failed for fold {fold_num}: {e}")

        onnx_save_path = os.path.join(PROJECT_FOLDER, f"efficientnet_b0_clean_fold_{fold_num}.onnx")
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
    metrics_save_path = os.path.join(PLOTS_FOLDER, "all_folds_detailed_metrics_clean.csv")
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\n✅ Detailed metrics for all folds saved to: {metrics_save_path}")


if __name__ == '__main__':
    print("This training module is not meant to be run directly. Please use 'train_cli.py' to launch training.")
    import sys as _sys
    _sys.exit(2)
