import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from base import COLORS, generate_rng, save_fig

MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

CORE_RADIUS = 2.0
SHELL_RADIUS = 5.0
RADIUS_STD = 0.4

GAUSSIAN_NAMES = ["Class A", "Class B"]
SHELL_NAMES = ["Class C (core)", "Class D (shell)"]

def generate_gaussians(rng):
    class_a = rng.multivariate_normal(MU_A, SIGMA_A, 500)
    class_b = rng.multivariate_normal(MU_B, SIGMA_B, 500)

    return np.vstack([class_a, class_b])

def generate_one_shell(rng, mean_radius):
    directions = rng.normal(size=(500, 5))
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / lengths

    radius = rng.normal(mean_radius, RADIUS_STD, (500, 1))

    return radius * unit_directions

def generate_shells(rng):
    core = generate_one_shell(rng, CORE_RADIUS)
    shell = generate_one_shell(rng, SHELL_RADIUS)

    return np.vstack([core, shell])

def center_distance(X, y):
    center_0 = X[y == 0].mean(axis=0)
    center_1 = X[y == 1].mean(axis=0)

    return np.linalg.norm(center_0 - center_1)

def plot_projection(ax, X, y, names, title):
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    for k in range(2):
        group = projected[y == k]
        ax.scatter(group[:, 0], group[:, 1], s=12, alpha=0.55, color=COLORS[k], label=names[k])

    ax.set_title(f"{title}\nPC1 + PC2 = {explained.sum() * 100:.1f}% of variance")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    return explained

def plot_radius_histogram(ax, X, y, names, title):
    radius = np.linalg.norm(X, axis=1)
    bins = np.linspace(radius.min(), radius.max(), 45)

    for k in range(2):
        ax.hist(radius[y == k], bins=bins, alpha=0.6, color=COLORS[k], label=names[k])

    ax.set_title(title)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

def print_radius_stats(X, y, names):
    radius = np.linalg.norm(X, axis=1)

    for k in range(2):
        group = radius[y == k]
        print(f"{names[k]}: radius mean = {group.mean():.4f}, std = {group.std():.4f}")

def main():
    rng = generate_rng()
    y = np.concatenate([[0] * 500, [1] * 500])

    # A and B
    print("\nA and B. Generating both datasets\n")

    gaussians = generate_gaussians(rng)
    shells = generate_shells(rng)

    print(f"Dataset I:  {gaussians.shape[0]} samples in {gaussians.shape[1]}D")
    print(f"Dataset II: {shells.shape[0]} samples in {shells.shape[1]}D")

    # C1
    print("\nC. PCA projection\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    explained_1 = plot_projection(axes[0], gaussians, y, GAUSSIAN_NAMES, "Dataset I: Gaussians")
    explained_2 = plot_projection(axes[1], shells, y, SHELL_NAMES, "Dataset II: Shells")
    fig.suptitle("Figure 4. PCA projection (5D -> 2D)")
    save_fig(fig, "fig4.png")

    print(f"Dataset I:  PC1 = {explained_1[0]:.4f}, PC2 = {explained_1[1]:.4f}, sum = {explained_1.sum():.4f}")
    print(f"Dataset II: PC1 = {explained_2[0]:.4f}, PC2 = {explained_2[1]:.4f}, sum = {explained_2.sum():.4f}")

    # C2
    print("\nDistance between class centers in 5D\n")

    print(f"Dataset I:  {center_distance(gaussians, y):.4f}")
    print(f"Dataset II: {center_distance(shells, y):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    plot_radius_histogram(axes[0], gaussians, y, GAUSSIAN_NAMES, "Dataset I: Gaussians")
    plot_radius_histogram(axes[1], shells, y, SHELL_NAMES, "Dataset II: Shells")
    fig.suptitle("Figure 5. Distribution of the radius of each point, by class")
    save_fig(fig, "fig5.png")

    print("\nRadius per class\n")

    print_radius_stats(gaussians, y, GAUSSIAN_NAMES)
    print_radius_stats(shells, y, SHELL_NAMES)

    # D
    print("\nD. A function of the inputs that separates Dataset II\n")

    squared_radius = (shells ** 2).sum(axis=1)
    threshold = ((CORE_RADIUS + SHELL_RADIUS) / 2) ** 2
    predicted = (squared_radius > threshold).astype(int)
    accuracy = np.mean(predicted == y)

    print(f"g(x) = sum(x_i^2) - {threshold:.2f}, the sign gives the class")
    print(f"accuracy of that single feature: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
