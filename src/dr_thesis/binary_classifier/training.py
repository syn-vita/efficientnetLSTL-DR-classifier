from __future__ import annotations

import argparse
import multiprocessing
import os
import warnings
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import models, transforms

from ..paths import CHECKPOINTS_DIR, WEBAPP_MODELS_DIR

DEFAULT_IMG_SIZE = 300
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_WEIGHTS_PATH = CHECKPOINTS_DIR / "fundus_classifier_efficientnet_best.pth"
DEFAULT_ONNX_OUT = WEBAPP_MODELS_DIR / "fundus_classifier_efficientnet_b3.onnx"


class SingleClassImageFolder(Dataset):
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        transform=None,
        label: int = 0,
        extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.label = int(label)
        self.extensions = {extension.lower() for extension in extensions}
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")

        self.samples = [
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        if not self.samples:
            raise FileNotFoundError(
                f"No image files with extensions {sorted(self.extensions)} found under {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label


def build_transforms(img_size: int) -> dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(img_size + 24),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "non_fundus": transforms.Compose(
            [
                transforms.Resize(img_size + 24),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)
    return model.to(device)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: dict[str, DataLoader],
    dataset_sizes: dict[str, int],
    device: torch.device,
    weights_path: Path,
    *,
    num_epochs: int,
) -> nn.Module:
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ("train", "val"):
            model.train(mode=(phase == "train"))
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = float(epoch_acc)
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")
    return model


def export_to_onnx(
    trained_model: nn.Module,
    weights_path: Path,
    onnx_out_path: Path,
    *,
    img_size: int,
) -> None:
    trained_model.eval()

    if weights_path.exists():
        try:
            state = torch.load(weights_path, map_location="cpu")
            trained_model.load_state_dict(state)
            print(f"Loaded best weights from '{weights_path}' for ONNX export.")
        except Exception as exc:
            warnings.warn(f"Could not load best weights: {exc}")
    else:
        warnings.warn("Best weights file not found; exporting current model state.")

    onnx_out_path.parent.mkdir(parents=True, exist_ok=True)
    trained_model_cpu = trained_model.to("cpu")
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    print(f"Exporting ONNX to: {onnx_out_path}")
    try:
        torch.onnx.export(
            trained_model_cpu,
            dummy_input,
            str(onnx_out_path),
            input_names=["input"],
            output_names=["logit"],
            dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=13,
            do_constant_folding=True,
        )
        print("ONNX export complete.")
        print("Note: Output is a single logit; apply sigmoid to obtain fundus probability.")
    except Exception as exc:
        warnings.warn(f"ONNX export failed: {exc}")


def build_dataloaders(
    fundus_dir: str,
    non_fundus_dir: str,
    device: torch.device,
    *,
    img_size: int,
    batch_size: int,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    transforms_map = build_transforms(img_size)

    print("Loading fundus images (single-class flat folder)...")
    fundus_dataset = SingleClassImageFolder(fundus_dir, transform=transforms_map["train"], label=0)

    print("Loading non-fundus images (single-class flat folder)...")
    non_fundus_dataset = SingleClassImageFolder(
        non_fundus_dir,
        transform=transforms_map["non_fundus"],
        label=1,
    )
    print(f"Non-fundus source: {non_fundus_dir}")
    print(f"Found {len(non_fundus_dataset)} non-fundus images.")

    fundus_count = len(fundus_dataset)
    non_fundus_count = len(non_fundus_dataset)
    if fundus_count == 0 or non_fundus_count == 0:
        raise RuntimeError("No images available after scanning datasets.")

    neg_target = min(non_fundus_count, 2 * fundus_count)

    def balanced_subset(dataset: Dataset, size: int):
        if len(dataset) <= size:
            return dataset
        subset, _ = random_split(dataset, [size, len(dataset) - size])
        return subset

    fundus_dataset_bal = balanced_subset(fundus_dataset, fundus_count)
    non_fundus_dataset_bal = balanced_subset(non_fundus_dataset, neg_target)
    ratio = len(non_fundus_dataset_bal) / max(1, len(fundus_dataset_bal))
    print(
        f"Balancing datasets: fundus={len(fundus_dataset_bal)}, "
        f"non_fundus={len(non_fundus_dataset_bal)} (~{ratio:.2f}:1)"
    )

    print("Combining datasets...")
    full_dataset = ConcatDataset([fundus_dataset_bal, non_fundus_dataset_bal])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_workers = min(8, (os.cpu_count() or 4))
    pin_memory = device.type == "cuda"
    persistent = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    print(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")
    return dataloaders, dataset_sizes


def prompt_path(message: str) -> str:
    while True:
        path_value = input(message).strip().strip('"')
        if path_value and os.path.isdir(path_value):
            return os.path.abspath(path_value)
        print("Invalid path. Please try again.\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the binary fundus-image classifier and export its ONNX model.",
    )
    parser.add_argument("--fundus-dir", help="Path to the folder containing fundus images.")
    parser.add_argument("--non-fundus-dir", help="Path to the folder containing non-fundus images.")
    parser.add_argument(
        "--weights-path",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to write the best .pth checkpoint. Default: artifacts/checkpoints/fundus_classifier_efficientnet_best.pth",
    )
    parser.add_argument(
        "--onnx-out",
        default=str(DEFAULT_ONNX_OUT),
        help="Path to write the exported ONNX model. Default: dr-classification-webapp/public/models/fundus_classifier_efficientnet_b3.onnx",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs. Default: {DEFAULT_EPOCHS}.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image size. Default: {DEFAULT_IMG_SIZE}.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the final confirmation prompt.")
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[str, str, Path, Path]:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.img_size < 32:
        raise ValueError("--img-size must be >= 32")

    fundus_dir = os.path.abspath(args.fundus_dir) if args.fundus_dir else prompt_path("Enter the path to the fundus image folder: ")
    non_fundus_dir = (
        os.path.abspath(args.non_fundus_dir)
        if args.non_fundus_dir
        else prompt_path("Enter the path to the non-fundus image folder: ")
    )
    if not os.path.isdir(fundus_dir):
        raise FileNotFoundError(f"Fundus directory not found: {fundus_dir}")
    if not os.path.isdir(non_fundus_dir):
        raise FileNotFoundError(f"Non-fundus directory not found: {non_fundus_dir}")

    return fundus_dir, non_fundus_dir, Path(args.weights_path).resolve(), Path(args.onnx_out).resolve()


def print_summary(
    fundus_dir: str,
    non_fundus_dir: str,
    weights_path: Path,
    onnx_out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    img_size: int,
) -> None:
    print("\nSummary:")
    print(f"  Fundus images:     {fundus_dir}")
    print(f"  Non-fundus images: {non_fundus_dir}")
    print(f"  Weights path:      {weights_path}")
    print(f"  ONNX output:       {onnx_out_path}")
    print(f"  Epochs:            {epochs}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Image size:        {img_size}")


def confirm_run(skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True
    confirm = input("\nProceed with training? [y/N]: ").strip().lower()
    return confirm in {"y", "yes"}


def run_training_workflow(
    fundus_dir: str,
    non_fundus_dir: str,
    weights_path: Path,
    onnx_out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    img_size: int,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataloaders, dataset_sizes = build_dataloaders(
        fundus_dir,
        non_fundus_dir,
        device,
        img_size=img_size,
        batch_size=batch_size,
    )
    print("Building model (EfficientNet-B3)...")
    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=DEFAULT_LEARNING_RATE)

    trained_model = train_model(
        model,
        criterion,
        optimizer,
        dataloaders,
        dataset_sizes,
        device,
        weights_path,
        num_epochs=epochs,
    )
    print("\nModel training finished.")
    print(f"Best model weights saved to '{weights_path}'")
    export_to_onnx(trained_model, weights_path, onnx_out_path, img_size=img_size)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fundus_dir, non_fundus_dir, weights_path, onnx_out_path = resolve_inputs(args)
    print_summary(
        fundus_dir,
        non_fundus_dir,
        weights_path,
        onnx_out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    if not confirm_run(args.yes):
        print("Aborted by user.")
        return 0
    return run_training_workflow(
        fundus_dir,
        non_fundus_dir,
        weights_path,
        onnx_out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except (RuntimeError, Exception):
        pass
    raise SystemExit(main())
