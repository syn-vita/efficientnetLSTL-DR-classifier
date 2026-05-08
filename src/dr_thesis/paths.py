from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
WEBAPP_DIR = REPO_ROOT / "dr-classification-webapp"
WEBAPP_MODELS_DIR = WEBAPP_DIR / "public" / "models"
