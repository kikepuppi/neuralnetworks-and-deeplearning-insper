import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from base import COLORS, generate_rng, plot_loss, save_fig
from mlp import MLP, accuracy, binary_cross_entropy, sigmoid

CENTERS = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
SIZES = [2, 8, 1]
ETA = 0.5
EPOCHS = 2000

def generate_data():
    X, c = make_blobs(n_samples=[500, 250, 250], centers=CENTERS, cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def plot_classes(ax, X, y):
    for k in range(2):
        cloud = X[y == k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=12, alpha=0.6, color=COLORS[k], label=f"Class {k}")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

def plot_regions(ax, model, X):
    grid_x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
    grid_y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    regions = model.predict(grid).reshape(mesh_x.shape)

    ax.contourf(mesh_x, mesh_y, regions, levels=[-0.5, 0.5, 1.5], colors=COLORS[:2], alpha=0.2)

def main():
    rng = generate_rng()

    # A
    print("\nA. Generate the data\n")

    X_train, X_test, y_train, y_test = generate_data()
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    print(f"train {X_train.shape}, test {X_test.shape}")
    print(f"class 1 share: train {y_train.mean():.4f}, test {y_test.mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_classes(ax, X_all, y_all)
    ax.set_title("Figure 1. Class 1 sits in two clusters on opposite sides of class 0")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig1.png")

    # B
    print("\nB. Linear baseline\n")

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"logistic regression test accuracy: {baseline.score(X_test, y_test):.4f}")

    # C
    print("\nC. MLP from scratch\n")

    model = MLP(SIZES, rng, ETA, sigmoid, binary_cross_entropy)
    model.train(X_train, y_train.reshape(-1, 1), EPOCHS)

    print(f"architecture {SIZES}, eta {ETA}, epochs {EPOCHS}, parameters {model.count_parameters()}")
    print(f"first training loss: {model.losses[0]:.4f}")
    print(f"final training loss: {model.losses[-1]:.4f}")
    print(f"test accuracy: {accuracy(model, X_test, y_test):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_loss(ax, model.losses, "Figure 2. Training loss per epoch")
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, "fig2.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_regions(ax, model, X_all)
    plot_classes(ax, X_test, y_test)
    ax.set_title("Figure 3. Decision regions over the test points")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig3.png")

if __name__ == "__main__":
    main()
