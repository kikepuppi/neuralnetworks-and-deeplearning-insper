from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
EXERCISE = Path(__file__).resolve().parent.parent
FIGURES = EXERCISE / "figures"
COLORS = ["#1f77b4", "#d62728"]
SAMPLES = 1000

def generate_rng():
    return np.random.default_rng(SEED)

def save_fig(fig, name):

    if not FIGURES.exists():
        FIGURES.mkdir()

    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")

def generate_points(rng, mean_0, mean_1, cov):
    # cov e a matriz de covariancia do enunciado, nao o desvio padrao
    class_0 = rng.multivariate_normal(mean_0, cov, SAMPLES)
    class_1 = rng.multivariate_normal(mean_1, cov, SAMPLES)

    X = np.vstack([class_0, class_1])
    y = np.concatenate([[0] * SAMPLES, [1] * SAMPLES])
    return X, y

def plot_classes(ax, X, y):
    for k in range(2):
        cloud = X[y == k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=10, alpha=0.5, color=COLORS[k], label=f"Class {k}")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

def plot_boundary(ax, w, b, X, label, color="black"):
    left = X[:, 0].min() - 0.5
    right = X[:, 0].max() + 0.5

    line_x = np.array([left, right])
    line_y = -(w[0] * line_x + b) / w[1]

    ax.plot(line_x, line_y, color=color, linewidth=2, label=label)
    ax.set_xlim(left, right)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

def plot_errors(ax, X, wrong):
    if wrong.sum() == 0:
        return

    ax.scatter(X[wrong, 0], X[wrong, 1], s=30, facecolors="none", edgecolors="black",
               linewidths=0.6, label=f"Misclassified ({wrong.sum()})")
