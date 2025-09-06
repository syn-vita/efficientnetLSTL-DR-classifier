# -*- coding: utf-8 -*-
import os
import time
import copy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- Dataset ---
class DRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.img_labels = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label


# --- Focal Loss (optionally weighted) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


def default_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25)
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }


def compute_class_weights(train_df: pd.DataFrame, num_classes: int, boost_weights: Dict[int, float] | None = None) -> Dict[int, float]:
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {}
    for cid in range(num_classes):
        if cid in class_counts:
            class_weights[cid] = float(total_samples) / (num_classes * float(class_counts[cid]))
        else:
            class_weights[cid] = 1.0
    if boost_weights:
        for cid, mult in boost_weights.items():
            if cid in class_weights:
                class_weights[cid] *= float(mult)
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
    train_dataset = DRDataset(train_df, image_dir, transforms_map['train'])
    val_dataset = DRDataset(val_df, image_dir, transforms_map['val'])

    sample_weights = [class_weights[label] for label in train_df['label']]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=(device.type == 'cuda')),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == 'cuda')),
    }
    sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
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
    best_bal_acc = 0.0
    epochs_no_improve = 0
    class_names = [str(i) for i in range(num_classes)]

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    metrics_log: List[Dict] = []
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            sampled_class_counts = torch.zeros(num_classes, dtype=torch.long) if phase == 'train' else None
            running_loss = 0.0
            running_corrects = 0
            all_labels, all_preds = [], []

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if phase == 'train':
                    sampled_class_counts += torch.bincount(labels.detach().cpu(), minlength=num_classes)
                else:
                    all_labels.extend(labels.cpu().numpy())

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'train':
                counts = {c: int(sampled_class_counts[c].item()) for c in range(num_classes)}
                print(f'Train samples per class (epoch {epoch+1}): {counts}')

            if phase == 'val':
                print("\nValidation Metrics:")
                balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                print(f"Balanced Accuracy: {balanced_acc:.4f}")
                report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
                cm = confusion_matrix(all_labels, all_preds)
                per_class_acc = cm.diagonal() / cm.sum(axis=1)
                for class_label in class_names:
                    cid = int(class_label)
                    acc = per_class_acc[cid] if cid < len(per_class_acc) else 0.0
                    metrics_log.append({
                        'fold': current_fold,
                        'epoch': epoch + 1,
                        'class': cid,
                        'accuracy': acc,
                        'precision': report_dict[class_label]['precision'],
                        'recall': report_dict[class_label]['recall'],
                        'f1-score': report_dict[class_label]['f1-score'],
                        'balanced_accuracy': balanced_acc,
                    })
                print("Per-Class Accuracy:")
                for i, acc in enumerate(per_class_acc):
                    print(f"  Class {i}: {acc:.4f}")
                print("-" * 25)

                if balanced_acc > best_bal_acc:
                    best_bal_acc = balanced_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"✨ New best balanced validation accuracy: {balanced_acc:.4f}. Saving model!")
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
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc (balanced): {best_bal_acc:4f}')
    model.load_state_dict(best_wts)
    return model, history, metrics_log


def plot_history(history: Dict[str, List[float]], fold_num: int, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Fold {fold_num} Training History', fontsize=16)
    ax1.plot(history['train_acc'], label='Train Acc')
    ax1.plot(history['val_acc'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.savefig(save_path)
    plt.close(fig)
