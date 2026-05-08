from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dr_thesis.models.cbam import CBAM, ChannelAttention, EfficientNetB0WithCBAM, SpatialAttention
from dr_thesis.training.cbam import run_training as main


if __name__ == "__main__":
    print("This training module is not meant to be run directly. Please use 'train_cli.py' to launch training.")
    raise SystemExit(2)
