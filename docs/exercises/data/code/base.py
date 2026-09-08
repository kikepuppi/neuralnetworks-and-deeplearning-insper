from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
EXERCISE = Path(__file__).resolve().parent.parent
FIGURES = EXERCISE / "figures"
DATA = EXERCISE / "data"
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

def generate_rng():
    return np.random.default_rng(SEED)

def save_fig(fig, name):

    if not FIGURES.exists():
        FIGURES.mkdir()

    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")