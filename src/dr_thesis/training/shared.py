from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm


class DRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.img_labels = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def default_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int,
    boost_weights: Dict[int, float] | None = None,
) -> Dict[int, float]:
    class_counts = train_df["label"].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights: Dict[int, float] = {}
    for class_id in range(num_classes):
        if class_id in class_counts:
            class_weights[class_id] = float(total_samples) / (num_classes * float(class_counts[class_id]))
        else:
            class_weights[class_id] = 1.0
    if boost_weights:
        for class_id, multiplier in boost_weights.items():
            if class_id in class_weights:
                class_weights[class_id] *= float(multiplier)
    return class_weights


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    transforms_map: Dict[str, transforms.Compose],
    batch_size: int,
    device: torch.device,
    num_workers: int,
    class_weights: Dict[int, float],
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    train_dataset = DRDataset(train_df, image_dir, transforms_map["train"])
    val_dataset = DRDataset(val_df, image_dir, transforms_map["val"])

    sample_weights = [class_weights[label] for label in train_df["label"]]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        ),
    }
    sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    return dataloaders, sizes


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    device: torch.device,
    current_fold: int,
    num_classes: int,
    num_epochs: int = 25,
    scheduler=None,
    patience: int = 10,
):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0
    class_names = [str(index) for index in range(num_classes)]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    metrics_log: List[Dict[str, float | int]] = []
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            sampled_class_counts = torch.zeros(num_classes, dtype=torch.long) if phase == "train" else None
            running_loss = 0.0
            running_corrects = 0
            all_labels: List[int] = []
            all_preds: List[int] = []

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if phase == "train":
                    sampled_class_counts += torch.bincount(labels.detach().cpu(), minlength=num_classes)
                else:
                    all_labels.extend(labels.cpu().numpy())

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == "val":
                    all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == "train" and sampled_class_counts is not None:
                counts = {class_id: int(sampled_class_counts[class_id].item()) for class_id in range(num_classes)}
                print(f"Train samples per class (epoch {epoch + 1}): {counts}")

            if phase == "val":
                print("\nValidation Metrics:")
                val_overall_acc = epoch_acc.item()
                print(f"Overall Accuracy: {val_overall_acc:.4f}")
                report_dict = classification_report(
                    all_labels,
                    all_preds,
                    target_names=class_names,
                    zero_division=0,
                    output_dict=True,
                )
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
                cm = confusion_matrix(all_labels, all_preds)
                per_class_acc = cm.diagonal() / cm.sum(axis=1)
                for class_label in class_names:
                    class_id = int(class_label)
                    acc = per_class_acc[class_id] if class_id < len(per_class_acc) else 0.0
                    metrics_log.append(
                        {
                            "fold": current_fold,
                            "epoch": epoch + 1,
                            "class": class_id,
                            "accuracy": acc,
                            "precision": report_dict[class_label]["precision"],
                            "recall": report_dict[class_label]["recall"],
                            "f1-score": report_dict[class_label]["f1-score"],
                            "overall_accuracy": val_overall_acc,
                        }
                    )
                print("Per-Class Accuracy:")
                for class_id, acc in enumerate(per_class_acc):
                    print(f"  Class {class_id}: {acc:.4f}")
                print("-" * 25)

                if val_overall_acc > best_val_acc:
                    best_val_acc = val_overall_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"New best overall validation accuracy: {val_overall_acc:.4f}. Saving model!")
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break

        if scheduler is not None:
            scheduler.step()
        print()
        if epochs_no_improve >= patience:
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc (overall): {best_val_acc:4f}")
    model.load_state_dict(best_wts)
    return model, history, metrics_log


def plot_history(history: Dict[str, List[float]], fold_num: int, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Fold {fold_num} Training History", fontsize=16)
    ax1.plot(history["train_acc"], label="Train Acc")
    ax1.plot(history["val_acc"], label="Val Acc")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.plot(history["train_loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.savefig(save_path)
    plt.close(fig)


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_num_workers() -> int:
    return max(2, (os.cpu_count() or 2) - 1)


def print_dataset_overview(all_labels_df: pd.DataFrame) -> None:
    print("\nDataset Overview:")
    print(f"Total samples: {len(all_labels_df)}")
    overall_class_counts = all_labels_df["label"].value_counts().sort_index()
    print(f"Class distribution: {overall_class_counts.to_dict()}")
    class_percentages = (overall_class_counts / len(all_labels_df) * 100).round(2)
    print(f"Class percentages: {class_percentages.to_dict()}")
    print("=" * 50)


def save_fold_artifacts(
    trained_model: nn.Module,
    fold_num: int,
    project_folder: str,
    model_filename_prefix: str,
    img_size: int,
) -> None:
    pth_save_path = os.path.join(project_folder, f"{model_filename_prefix}_fold_{fold_num}.pth")
    try:
        torch.save(trained_model.state_dict(), pth_save_path)
        print(f"Best model for fold {fold_num} saved to: {pth_save_path}")
    except Exception as exc:
        print(f"[ERROR] Torch checkpoint save failed for fold {fold_num}: {exc}")

    onnx_save_path = os.path.join(project_folder, f"{model_filename_prefix}_fold_{fold_num}.onnx")
    try:
        export_model = trained_model.to("cpu").eval()
        dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")
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
    except Exception as exc:
        print(f"[ERROR] ONNX export failed for fold {fold_num}: {exc}")


def finalize_metrics(
    all_folds_metrics: List[Dict[str, float | int]],
    plots_folder: str,
    metrics_filename: str,
) -> None:
    print("\nAll folds trained successfully!")
    metrics_df = pd.DataFrame(all_folds_metrics)
    metrics_save_path = os.path.join(plots_folder, metrics_filename)
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\nDetailed metrics for all folds saved to: {metrics_save_path}")


def run_kfold_training(
    *,
    image_dir: str,
    annotations_csv: str,
    project_folder: str,
    plots_folder: str,
    model_key: str,
    model_filename_prefix: str,
    metrics_filename: str,
    batch_size: int,
    img_size: int,
    num_epochs: int,
    num_classes: int,
    n_splits: int,
    device: torch.device,
    num_workers: int,
    use_pretrained: bool,
    boost_weights: Dict[int, float] | None,
    criterion_builder: Callable[[nn.Module, Dict[int, float], torch.device], nn.Module],
    optimizer_builder: Callable[[nn.Module], torch.optim.Optimizer],
    scheduler_builder: Callable[[torch.optim.Optimizer, int], Any],
    fold_seed_callback: Callable[[], None],
    load_records: Callable[[str], pd.DataFrame],
    model_builder: Callable[..., nn.Module],
    patience: int = 7,
) -> None:
    os.makedirs(plots_folder, exist_ok=True)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found at: {image_dir}")
    if not os.path.isfile(annotations_csv):
        raise FileNotFoundError(f"Annotations CSV not found at: {annotations_csv}")

    print(f"Using image folder: {image_dir}")
    print(f"Using annotations CSV: {annotations_csv}")
    print(f"Outputs will be saved under: {project_folder}")

    all_labels_df = load_records(annotations_csv)
    print(f"Loaded {len(all_labels_df)} records from {annotations_csv}")

    transforms_map = default_transforms(img_size)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    x_values = all_labels_df["image_filename"]
    y_values = all_labels_df["label"]
    print_dataset_overview(all_labels_df)

    all_folds_metrics: List[Dict[str, float | int]] = []
    for fold_num, (train_index, val_index) in enumerate(skf.split(x_values, y_values), 1):
        print("\n" + "=" * 50)
        print(f"STARTING FOLD {fold_num}/{n_splits}")
        print("=" * 50)

        fold_seed_callback()

        train_df = all_labels_df.iloc[train_index].reset_index(drop=True)
        val_df = all_labels_df.iloc[val_index].reset_index(drop=True)

        class_weights = compute_class_weights(train_df, num_classes, boost_weights=boost_weights)
        print(f"Class counts in training set for Fold {fold_num}: {train_df['label'].value_counts().sort_index().to_dict()}")
        print(f"Class weights: {class_weights}")

        dataloaders, dataset_sizes = make_dataloaders(
            train_df,
            val_df,
            image_dir,
            transforms_map,
            batch_size,
            device,
            num_workers,
            class_weights,
        )

        model = model_builder(model_key, num_classes=num_classes, device=device, use_pretrained=use_pretrained)
        criterion = criterion_builder(model, class_weights, device)
        optimizer = optimizer_builder(model)
        scheduler = scheduler_builder(optimizer, num_epochs)

        trained_model, history, fold_metrics = train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            dataset_sizes,
            device,
            current_fold=fold_num,
            num_classes=num_classes,
            num_epochs=num_epochs,
            scheduler=scheduler,
            patience=patience,
        )
        all_folds_metrics.extend(fold_metrics)

        save_fold_artifacts(
            trained_model=trained_model,
            fold_num=fold_num,
            project_folder=project_folder,
            model_filename_prefix=model_filename_prefix,
            img_size=img_size,
        )

        plot_save_path = os.path.join(plots_folder, f"Fold{fold_num}_Figure.png")
        plot_history(history, fold_num, plot_save_path)
        print(f"Training history plot for fold {fold_num} saved to: {plot_save_path}")

    finalize_metrics(all_folds_metrics, plots_folder, metrics_filename)
