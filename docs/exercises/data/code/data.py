"""Exercise: data.

Runnable from a clean clone:
    python docs/exercises/data/code/data.py
Figures are written to docs/exercises/data/figures/.
"""

from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
