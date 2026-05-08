from __future__ import annotations

try:
    from ._bootstrap import add_src_to_path
except ImportError:
    from _bootstrap import add_src_to_path

add_src_to_path()

from dr_thesis.evaluation.folds import main


if __name__ == "__main__":
    raise SystemExit(main())
