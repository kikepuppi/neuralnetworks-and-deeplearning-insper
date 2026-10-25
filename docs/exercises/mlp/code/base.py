from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
EXERCISE = Path(__file__).resolve().parent.parent
FIGURES = EXERCISE / "figures"
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

def generate_rng(seed=SEED):
    return np.random.default_rng(seed)

def save_fig(fig, name):

    if not FIGURES.exists():
        FIGURES.mkdir()

    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")

def plot_loss(ax, losses, title):
    ax.plot(losses, color=COLORS[0], label="Training loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
