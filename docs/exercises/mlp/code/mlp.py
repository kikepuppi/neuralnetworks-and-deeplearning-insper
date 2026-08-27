"""Exercise: mlp.

Runnable from a clean clone:
    python docs/exercises/mlp/code/mlp.py
Figures are written to docs/exercises/mlp/figures/.
"""

from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
