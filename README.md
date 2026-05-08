# Diabetic Retinopathy Classification

Primary documentation for the repository. This project contains the Python workflows for diabetic retinopathy model training, evaluation, and ONNX export, plus the React web app that serves browser-based inference.

## Prerequisites

- Windows with PowerShell
- Python 3.10+ and `pip`
- Node.js 18+
- A local Python virtual environment for the training and evaluation workflows

Create and activate the virtual environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

This repo does not currently track a Python dependency manifest. The refactor smoke checks were verified with:
`torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pillow`, `tqdm`, `onnx`, `onnxruntime`, `thop`, and `fvcore`.

For the frontend:

```powershell
cd dr-classification-webapp
npm install
```

## Project Structure

- `scripts/` - canonical commands you run from the repo root
- `src/dr_thesis/` - maintained Python implementation
- `src/dr_thesis/training/` - DR model training logic
- `src/dr_thesis/evaluation/` - fold evaluation logic
- `src/dr_thesis/export/` - ONNX export logic
- `src/dr_thesis/binary_classifier/` - fundus-image classifier workflow used by the web app
- `data/` - tracked CSV metadata for the active DR workflows
- `data/legacy/` - older thesis metadata retained for reproduction
- `artifacts/checkpoints/` - `.pth` checkpoints
- `artifacts/evaluation/` - evaluation outputs
- `artifacts/exports/` - exported ONNX files
- `artifacts/archive/` - historical thesis outputs moved out of the active workflow
- `docs/references/` - thesis papers and reference PDFs
- `dr-classification-webapp/` - Vite/React frontend
- `dr-classification-webapp/public/models/` - ONNX models loaded by the browser app

## Data Preparation

### DR model training data

The DR training workflow expects:
- an image directory you supply with `--image-dir`
- a CSV labels file you supply with `--csv-path`

The CSV should match the APTOS-style format used by the training code, with columns like:

```text
id_code,diagnosis
```

The training loader converts `id_code` values into image filenames by appending `.png`, so your image folder should contain files like:

```text
<id_code>.png
```

If you are using the same dataset source as the earlier thesis workflow, the original resized APTOS image download referenced by the project was:
https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data?select=colored_images

Tracked CSVs already in the repo:
- `data/dataset.csv`
- `data/train.csv`
- `data/valid.csv`
- `data/test.csv`

### DR evaluation data

The evaluator expects:
- an evaluation image directory
- a labels file passed with `--labels-path`

Supported label file formats:
- `.csv`
- `.tsv`
- `.dat`

The default tracked test labels file is `data/test.csv`.

### Fundus classifier data

The binary fundus-image classifier expects two image folders:
- `--fundus-dir` for retinal fundus images
- `--non-fundus-dir` for non-retinal images

Both folders can be flat or recursive. Supported image extensions include:
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

### Legacy metadata

Historical metadata retained for reproduction:
- `data/legacy/aptos2019_labels.csv`
- `data/legacy/messidor_data.csv`

## Canonical Python Commands

### Train DR models

Show options:

```powershell
python scripts/train.py --help
```

Example:

```powershell
python scripts/train.py --model baseline --image-dir <path-to-images> --csv-path data/train.csv --out-dir artifacts/checkpoints --yes
```

Notes:
- `--model` accepts `baseline`, `cbam`, `lstl`, or `all`
- if `--out-dir` is omitted, the script writes outputs to the CSV folder
- for best compatibility with the evaluation and export defaults, use `artifacts/checkpoints/` explicitly

### Train the fundus-image classifier

Show options:

```powershell
python scripts/train_fundus_classifier.py --help
```

Example:

```powershell
python scripts/train_fundus_classifier.py --fundus-dir <path-to-fundus-images> --non-fundus-dir <path-to-non-fundus-images> --yes
```

Default outputs:
- best checkpoint: `artifacts/checkpoints/fundus_classifier_efficientnet_best.pth`
- ONNX export: `dr-classification-webapp/public/models/fundus_classifier_efficientnet_b3.onnx`

### Evaluate DR checkpoints

Show options:

```powershell
python scripts/evaluate.py --help
```

Example:

```powershell
python scripts/evaluate.py --models baseline --images-dir <path-to-eval-images> --labels-path data/test.csv --models-dir artifacts/checkpoints --out-dir artifacts/evaluation
```

Default paths:
- checkpoint folder: `artifacts/checkpoints/`
- output folder: `artifacts/evaluation/`

### Export DR ONNX models

Preview planned exports:

```powershell
python scripts/export_onnx.py --print-only
```

Export to the default artifact location:

```powershell
python scripts/export_onnx.py --pth-dir artifacts/checkpoints --out-dir artifacts/exports
```

Export directly into the web app model folder:

```powershell
python scripts/export_onnx.py --pth-dir artifacts/checkpoints --out-dir dr-classification-webapp/public/models
```

## Web App

Start the frontend:

```powershell
cd dr-classification-webapp
npm run dev
```

Other frontend commands:

```powershell
npm run build
npm run lint
npm run preview
```

The app currently expects these ONNX files in `dr-classification-webapp/public/models/`:
- `/models/efficientnet_b0_clean_fold_3.onnx`
- `/models/efficientnet_b0_cbam_clean_fold_4.onnx`
- `/models/efficientnet_b0_lstl_clean_fold_4.onnx`
- `/models/fundus_classifier_efficientnet_b3.onnx`

Where they are wired:
- DR model paths: `dr-classification-webapp/src/config/models.js`
- fundus classifier path: `dr-classification-webapp/src/components/ImageUpload.jsx`

## Output Locations

Current default output behavior:

- DR training:
  - `.pth` and `.onnx` files go to the `--out-dir` you choose
  - training figures go under `Figure Outputs/` inside that same output location
- DR evaluation:
  - outputs go to `artifacts/evaluation/` by default
- DR export:
  - ONNX files go to `artifacts/exports/` by default
- fundus classifier:
  - checkpoint goes to `artifacts/checkpoints/`
  - ONNX goes to `dr-classification-webapp/public/models/`

## Verification

Minimum safe verification after structural changes:

```powershell
python scripts/smoke_check.py
python scripts/train.py --help
python scripts/train_fundus_classifier.py --help
python scripts/evaluate.py --help
python scripts/export_onnx.py --print-only
```
