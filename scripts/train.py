from __future__ import annotations

import argparse
import os
from typing import Callable

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from dr_thesis.training import baseline, cbam, lstl

Trainer = Callable[[str, str, str | None], None]

TRAINERS: dict[str, Trainer] = {
    "baseline": baseline.run_training,
    "cbam": cbam.run_training,
    "lstl": lstl.run_training,
}

MODEL_NAMES = {
    "baseline": "EfficientNet-B0 (baseline)",
    "cbam": "EfficientNet-B0 + CBAM",
    "lstl": "EfficientNet-B0 + LSTL",
    "all": "ALL (runs baseline -> cbam -> lstl)",
}

MODEL_MENU = {
    "1": ("baseline", MODEL_NAMES["baseline"]),
    "2": ("cbam", MODEL_NAMES["cbam"]),
    "3": ("lstl", MODEL_NAMES["lstl"]),
    "4": ("all", "Train ALL (1, 2, 3)"),
}


def prompt_choice(prompt: str) -> str:
    while True:
        print(prompt)
        for key, (_, description) in MODEL_MENU.items():
            print(f"  {key}) {description}")
        selection = input("> ").strip()
        if selection.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            raise SystemExit(0)
        if selection in MODEL_MENU:
            return MODEL_MENU[selection][0]
        print("Invalid selection. Try again (or 'q' to quit).\n")


def prompt_path(message: str, check_fn) -> str:
    while True:
        path_value = input(message).strip().strip('"')
        if path_value and check_fn(path_value):
            return os.path.abspath(path_value)
        print("Invalid path. Please try again.\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 diabetic retinopathy models.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "cbam", "lstl", "all"],
        help="Model variant to train. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--image-dir",
        help="Path to the folder containing training images.",
    )
    parser.add_argument(
        "--csv-path",
        help="Path to the CSV file containing image labels.",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory to write checkpoints and plots. Defaults to the CSV folder.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt once inputs are resolved.",
    )
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[str, str, str, str | None]:
    interactive_mode = any(value is None for value in (args.model, args.image_dir, args.csv_path))
    model_key = args.model or prompt_choice("Select the model to train:")
    image_dir = os.path.abspath(args.image_dir) if args.image_dir else prompt_path(
        "Enter the path to the image folder: ",
        os.path.isdir,
    )
    csv_path = os.path.abspath(args.csv_path) if args.csv_path else prompt_path(
        "Enter the path to the CSV file: ",
        os.path.isfile,
    )

    if args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
    elif interactive_mode:
        raw_out_dir = input(
            "Enter the output directory (press Enter to use the CSV's folder): "
        ).strip().strip('"')
        out_dir = os.path.abspath(raw_out_dir) if raw_out_dir else None
    else:
        out_dir = None

    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    return model_key, image_dir, csv_path, out_dir


def print_summary(model_key: str, image_dir: str, csv_path: str, out_dir: str | None) -> None:
    print("\nSummary:")
    print(f"  Model: {MODEL_NAMES[model_key]}")
    print(f"  Images: {image_dir}")
    print(f"  CSV:    {csv_path}")
    print(f"  Output: {out_dir if out_dir else os.path.dirname(csv_path)}")


def confirm_run(skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True
    confirm = input("\nProceed with training? [y/N]: ").strip().lower()
    return confirm in {"y", "yes"}


def run_selected_training(model_key: str, image_dir: str, csv_path: str, out_dir: str | None) -> None:
    if model_key == "all":
        for key in ("baseline", "cbam", "lstl"):
            print(f"\n=== Running {MODEL_NAMES[key]} ===\n")
            TRAINERS[key](image_dir=image_dir, csv_path=csv_path, out_dir=out_dir)
        print("\nAll trainings completed.\n")
        return

    TRAINERS[model_key](image_dir=image_dir, csv_path=csv_path, out_dir=out_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_key, image_dir, csv_path, out_dir = resolve_inputs(args)
    print_summary(model_key, image_dir, csv_path, out_dir)

    if not confirm_run(args.yes):
        print("Aborted by user.")
        return 0

    print("\nStarting training...\n")
    run_selected_training(model_key, image_dir, csv_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
