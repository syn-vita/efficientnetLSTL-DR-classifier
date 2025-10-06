# -*- coding: utf-8 -*-
import os
import sys
import importlib.util


def prompt_choice(prompt: str, choices: dict[str, str]) -> str:
    """Prompt user to choose a key from choices and return the selected key.

    choices: mapping from user-visible key (e.g., '1') to description.
    """
    while True:
        print(prompt)
        for k, desc in choices.items():
            print(f"  {k}) {desc}")
        sel = input("> ").strip()
        if sel.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            sys.exit(0)
        if sel in choices:
            return sel
        print("Invalid selection. Try again (or 'q' to quit).\n")


def prompt_path(msg: str, check_fn) -> str:
    """Prompt until a valid path passes the given check_fn (os.path.isdir or os.path.isfile)."""
    while True:
        p = input(msg).strip().strip('"')
        if p and check_fn(p):
            return os.path.abspath(p)
        print("Invalid path. Please try again.\n")


def main():
    # Determine paths to training scripts (same directory as this file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = {
        "1": os.path.join(script_dir, "train_efficientnet_b0.py"),
        "2": os.path.join(script_dir, "train_efficientnet_b0_cbam.py"),
        "3": os.path.join(script_dir, "train_efficientnet_b0_lstl.py"),
    }
    names = {
        "1": "EfficientNet-B0 (baseline)",
        "2": "EfficientNet-B0 + CBAM",
        "3": "EfficientNet-B0 + LSTL",
    }

    choice = prompt_choice(
        "Select the model to train:",
        {"1": names["1"], "2": names["2"], "3": names["3"]},
    )
    script_path = scripts[choice]
    if not os.path.isfile(script_path):
        print(f"Training script not found: {script_path}")
        sys.exit(1)

    print(f"\nSelected: {names[choice]}\n")

    image_dir = prompt_path("Enter the path to the image folder: ", os.path.isdir)
    csv_path = prompt_path("Enter the path to the CSV file: ", os.path.isfile)
    out_dir_raw = input("Enter the output directory (press Enter to use the CSV's folder): ").strip().strip('"')
    out_dir = os.path.abspath(out_dir_raw) if out_dir_raw else None
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("\nSummary:")
    print(f"  Model: {names[choice]}")
    print(f"  Images: {image_dir}")
    print(f"  CSV:    {csv_path}")
    print(f"  Output: {out_dir if out_dir else os.path.dirname(csv_path)}")

    confirm = input("\nProceed with training? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Aborted by user.")
        return

    print("\nStarting training...\n")
    # Dynamically import the selected training module and call main(image_dir, csv_path, out_dir)
    spec = importlib.util.spec_from_file_location("trainer_module", script_path)
    if spec is None or spec.loader is None:
        print("Failed to load trainer module.")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "main"):
        print("Trainer module does not define a main(image_dir, csv_path, out_dir) function.")
        sys.exit(1)
    # Call the trainer
    mod.main(image_dir=image_dir, csv_path=csv_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
