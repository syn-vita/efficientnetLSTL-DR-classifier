from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_training_records(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    try:
        base_df = pd.read_csv(csv_path)
        if {"id_code", "diagnosis"}.issubset(base_df.columns):
            df = base_df[["id_code", "diagnosis"]].copy()
        else:
            df = pd.read_csv(csv_path, usecols=[0, 1])
            df.columns = ["id_code", "diagnosis"]
    except Exception:
        df = pd.read_csv(csv_path, usecols=["id_code", "diagnosis"])

    df["id_code"] = df["id_code"].astype(str)
    df = df.rename(columns={"id_code": "image_filename", "diagnosis": "label"})
    df["image_filename"] = df["image_filename"] + ".png"
    return df


def read_eval_labels(labels_path: str | Path) -> List[Tuple[str, int]]:
    labels_path = Path(labels_path)
    suffix = labels_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        if suffix == ".csv":
            df = pd.read_csv(labels_path)
        else:
            df = pd.read_csv(labels_path, sep="\t")

        cols = [column.lower() for column in df.columns]
        if "image_filename" in cols and "label" in cols:
            image_col = df.columns[cols.index("image_filename")]
            label_col = df.columns[cols.index("label")]
        elif "id_code" in cols and "diagnosis" in cols:
            image_col = df.columns[cols.index("id_code")]
            label_col = df.columns[cols.index("diagnosis")]
        else:
            image_col, label_col = df.columns[:2]
        return [
            (str(image_name), int(label))
            for image_name, label in df[[image_col, label_col]].itertuples(index=False, name=None)
        ]

    rows: List[Tuple[str, int]] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], int(parts[1])))
    return rows
