from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..data.records import load_training_records
from ..models.factory import build_model
from .shared import FocalLoss, default_device, default_num_workers, run_kfold_training


def run_training(image_dir: str, csv_path: str, out_dir: Optional[str] = None) -> None:
    image_dir = os.path.abspath(image_dir)
    annotations_csv = os.path.abspath(csv_path)
    project_folder = os.path.abspath(out_dir) if out_dir else os.path.dirname(annotations_csv)
    plots_folder = os.path.join(project_folder, "Figure Outputs", "EfficientNet-B0-LSTL-clean")

    device = default_device()
    batch_size = 32
    img_size = 224
    num_epochs = 60
    num_classes = 5
    lr_backbone = 3e-5
    lr_head = 1e-4
    weight_decay = 1e-4
    n_splits = 5
    num_workers = default_num_workers()

    use_pretrained = True
    use_focal_loss = True
    focal_gamma = 2.0
    boost_weights = None

    def fold_seed_callback() -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def criterion_builder(_model: nn.Module, _class_weights, _device: torch.device) -> nn.Module:
        if use_focal_loss:
            return FocalLoss(alpha=None, gamma=focal_gamma)
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def optimizer_builder(model: nn.Module) -> torch.optim.Optimizer:
        backbone_params = list(model.base.features.parameters())
        head_params = list(model.lstl.parameters()) + list(model.base.classifier.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )

    def scheduler_builder(optimizer: torch.optim.Optimizer, total_epochs: int):
        warmup_epochs = 3
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6),
            ],
            milestones=[warmup_epochs],
        )

    run_kfold_training(
        image_dir=image_dir,
        annotations_csv=annotations_csv,
        project_folder=project_folder,
        plots_folder=plots_folder,
        model_key="lstl",
        model_filename_prefix="efficientnet_b0_lstl_clean",
        metrics_filename="all_folds_detailed_metrics_lstl_clean.csv",
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
