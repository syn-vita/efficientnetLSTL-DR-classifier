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

    menu = {"1": names["1"], "2": names["2"], "3": names["3"], "4": "Train ALL (1, 2, 3)"}
    choice = prompt_choice("Select the model to train:", menu)
    if choice != "4":
        script_path = scripts[choice]
        if not os.path.isfile(script_path):
            print(f"Training script not found: {script_path}")
            sys.exit(1)
        print(f"\nSelected: {names[choice]}\n")
    else:
        print("\nSelected: Train ALL models (baseline, CBAM, LSTL)\n")

    image_dir = prompt_path("Enter the path to the image folder: ", os.path.isdir)
    csv_path = prompt_path("Enter the path to the CSV file: ", os.path.isfile)
    out_dir_raw = input("Enter the output directory (press Enter to use the CSV's folder): ").strip().strip('"')
    out_dir = os.path.abspath(out_dir_raw) if out_dir_raw else None
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("\nSummary:")
    if choice == "4":
        print("  Model: ALL (runs 1 -> 2 -> 3)")
    else:
        print(f"  Model: {names[choice]}")
    print(f"  Images: {image_dir}")
    print(f"  CSV:    {csv_path}")
    print(f"  Output: {out_dir if out_dir else os.path.dirname(csv_path)}")

    confirm = input("\nProceed with training? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Aborted by user.")
        return

    print("\nStarting training...\n")
    def _run_trainer(script_path_: str):
        spec_ = importlib.util.spec_from_file_location("trainer_module", script_path_)
        if spec_ is None or spec_.loader is None:
            print(f"Failed to load trainer module: {script_path_}")
            sys.exit(1)
        mod_ = importlib.util.module_from_spec(spec_)
        sys.modules["trainer_module"] = mod_
        spec_.loader.exec_module(mod_)  # type: ignore[attr-defined]
        if not hasattr(mod_, "main"):
            print("Trainer module does not define a main(image_dir, csv_path, out_dir) function.")
            sys.exit(1)
        mod_.main(image_dir=image_dir, csv_path=csv_path, out_dir=out_dir)

    if choice == "4":
        # Run all three trainers in order
        for key in ["1", "2", "3"]:
            print(f"\n=== Running {names[key]} ===\n")
            _run_trainer(scripts[key])
        print("\nAll trainings completed.\n")
    else:
        _run_trainer(script_path)


if __name__ == "__main__":
    main()
