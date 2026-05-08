from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..data.records import load_training_records
from ..models.factory import build_model
from .shared import FocalLoss, default_device, default_num_workers, run_kfold_training


def run_training(image_dir: str, csv_path: str, out_dir: Optional[str] = None) -> None:
    image_dir = os.path.abspath(image_dir)
    annotations_csv = os.path.abspath(csv_path)
    project_folder = os.path.abspath(out_dir) if out_dir else os.path.dirname(annotations_csv)
    plots_folder = os.path.join(project_folder, "Figure Outputs", "EfficientNet-B0-CBAM-clean")

    device = default_device()
    batch_size = 32
    img_size = 224
    num_epochs = 60
    num_classes = 5
    learning_rate = 1e-4
    weight_decay = 1e-4
    n_splits = 5
    num_workers = default_num_workers()

    use_pretrained = True
    use_focal_loss = True
    focal_gamma = 2.0
    boost_weights = {2: 1.8}

    def fold_seed_callback() -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def criterion_builder(_model: nn.Module, class_weights, device_: torch.device) -> nn.Module:
        class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device_)
        if use_focal_loss:
            return FocalLoss(alpha=class_weights_tensor, gamma=focal_gamma)
        return nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    def optimizer_builder(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def scheduler_builder(optimizer: torch.optim.Optimizer, total_epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    run_kfold_training(
        image_dir=image_dir,
        annotations_csv=annotations_csv,
        project_folder=project_folder,
        plots_folder=plots_folder,
        model_key="cbam",
        model_filename_prefix="efficientnet_b0_cbam_clean",
        metrics_filename="all_folds_detailed_metrics_cbam_clean.csv",
        batch_size=batch_size,
        img_size=img_size,
        num_epochs=num_epochs,
        num_classes=num_classes,
        n_splits=n_splits,
        device=device,
        num_workers=num_workers,
        use_pretrained=use_pretrained,
        boost_weights=boost_weights,
        criterion_builder=criterion_builder,
        optimizer_builder=optimizer_builder,
        scheduler_builder=scheduler_builder,
        fold_seed_callback=fold_seed_callback,
        load_records=load_training_records,
        model_builder=build_model,
        patience=7,
    )
