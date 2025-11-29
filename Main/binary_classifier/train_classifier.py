import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, models, transforms
from PIL import Image
import os
from pathlib import Path
import warnings
import multiprocessing

# --- 1. CONFIGURATION ---
FUNDUS_IMG_PATH = r"C:\Users\Luigi\Desktop\code\Thesis\Main\APTOS 2019"
NUM_FUNDUS_IMAGES = 3662
NON_FUNDUS_ROOT = r"C:\Users\Luigi\Downloads\archive(1)\train"
IMG_SIZE = 300 # EfficientNet-B3 optimal input size
BATCH_SIZE = 16 # Reduce if you run out of memory
LEARNING_RATE = 0.0001
EPOCHS = 20
BEST_WEIGHTS_PATH = Path('fundus_classifier_efficientnet_best.pth')


# --- 3. DATA TRANSFORMATION & AUGMENTATION ---
# Define transforms for training (with augmentation) and validation (without)
# Normalization values are specific for models pre-trained on ImageNet
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE + 24), # Resize to a slightly larger size
        transforms.CenterCrop(IMG_SIZE),  # Crop to the final size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Lighter transform for non-fundus negatives to reduce CPU work
non_fundus_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 24),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SingleClassImageFolder(torch.utils.data.Dataset):
    """A simple dataset for a flat folder of images belonging to a single class.
    Returns label for every image as the provided `label` (default 0).
    """
    def __init__(self, root: str | os.PathLike, transform=None, label: int = 0,
                 extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        self.root = Path(root)
        self.transform = transform
        self.label = int(label)
        self.extensions = set(e.lower() for e in extensions)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")
        # Collect all image files recursively
        self.samples = [p for p in self.root.rglob('*') if p.is_file() and p.suffix.lower() in self.extensions]
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No image files with extensions {sorted(self.extensions)} found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
                print(f'New best model saved with accuracy: {best_acc:.4f}')

    print(f'\nTraining complete! Best Val Acc: {best_acc:4f}')
    return model


def export_to_onnx(trained_model: nn.Module, img_size: int = IMG_SIZE):
    """
    Export the trained binary EfficientNet-B3 classifier to ONNX.

    - Input:  (1, 3, img_size, img_size)
    - Output: (1, 1) raw logit (apply sigmoid at inference time for probability)
    """
    trained_model.eval()

    # Load best weights for export if available
    if BEST_WEIGHTS_PATH.exists():
        try:
            state = torch.load(BEST_WEIGHTS_PATH, map_location='cpu')
            trained_model.load_state_dict(state)
            print(f"Loaded best weights from '{BEST_WEIGHTS_PATH}' for ONNX export.")
        except Exception as e:
            warnings.warn(f"Could not load best weights: {e}")
    else:
        warnings.warn("Best weights file not found; exporting current model state.")

    # Move to CPU for portable export
    trained_model_cpu = trained_model.to('cpu')
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    # Determine repo root (Thesis/) and output path under webapp public/models
    repo_root = Path(__file__).resolve().parents[2]
    web_models_dir = repo_root / 'dr-classification-webapp' / 'public' / 'models'
    web_models_dir.mkdir(parents=True, exist_ok=True)
    onnx_out_path = web_models_dir / 'fundus_classifier_efficientnet_b3.onnx'

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
    except Exception as e:
        warnings.warn(f"ONNX export failed: {e}")


def main():
    # --- 2. SETUP DEVICE (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # speed up convs for fixed input sizes

    # --- 4. LOAD DATASETS ---

    # a) Load positive examples (Fundus Images)
    print("Loading fundus images (single-class flat folder)...")
    fundus_dataset = SingleClassImageFolder(FUNDUS_IMG_PATH, transform=data_transforms['train'], label=0)

    # b) Load negative examples (Non-Fundus Images) from local folder
    print("Loading non-fundus images (local dataset)...")
    non_fundus_dataset = SingleClassImageFolder(NON_FUNDUS_ROOT, transform=non_fundus_transform, label=1)
    print(f"Non-fundus source: {NON_FUNDUS_ROOT}")
    print(f"Found {len(non_fundus_dataset)} non-fundus images.")

    # Balance so that non-fundus has 2x the number of fundus images (capped by availability)
    n_fundus = len(fundus_dataset)
    n_non = len(non_fundus_dataset)
    if n_fundus == 0 or n_non == 0:
        raise RuntimeError("No images available after scanning datasets.")
    neg_target = min(n_non, 2 * n_fundus)

    def balanced_subset(dataset, size):
        if len(dataset) <= size:
            return dataset
        subset, _ = random_split(dataset, [size, len(dataset) - size])
        return subset

    fundus_dataset_bal = balanced_subset(fundus_dataset, n_fundus)  # keep all fundus
    non_fundus_dataset_bal = balanced_subset(non_fundus_dataset, neg_target)  # 2x fundus, if available
    ratio = len(non_fundus_dataset_bal) / max(1, len(fundus_dataset_bal))
    print(f"Balancing datasets: fundus={len(fundus_dataset_bal)}, non_fundus={len(non_fundus_dataset_bal)} (~{ratio:.2f}:1)")

    # c) Combine into a single dataset (class ratio ~1:2 fundus:non-fundus)
    print("Combining datasets...")
    full_dataset = ConcatDataset([fundus_dataset_bal, non_fundus_dataset_bal])

    # d) Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # e) Create DataLoaders
    NUM_WORKERS = min(8, (os.cpu_count() or 4))
    PIN_MEMORY = device.type == 'cuda'
    PERSISTENT = NUM_WORKERS > 0
    PREFETCH = 2

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT,
            prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT,
            prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        ),
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")

    # --- 5. BUILD THE MODEL ---
    print("Building model (EfficientNet-B3)...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Freeze all the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer for our binary task
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)  # Output is 1 for binary classification

    model = model.to(device)

    # --- 6. DEFINE LOSS AND OPTIMIZER ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # --- 7. TRAIN ---
    model_ft = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=EPOCHS)
    print("\nModel training finished.")
    print(f"Best model weights saved to '{BEST_WEIGHTS_PATH}'")

    # --- 8. EXPORT TO ONNX ---
    export_to_onnx(model_ft, IMG_SIZE)


if __name__ == '__main__':
    # Windows multiprocessing safety for DataLoader spawn
    multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, Exception):
        pass
    main()