import matplotlib.pyplot as plt
import numpy as np
from base import COLORS, generate_rng, save_fig

MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
SCALES = [0.5, 1.0, 2.0, 4.0]

def generate_points(rng, scale):
    points = []
    labels = []

    for k in range(4):
        points.append(rng.normal(MEANS[k], STDS[k] * scale, (100, 2)))
        labels.append([k]*100)

    return np.vstack(points), np.concatenate(labels)

def plot_distributions(ax, X, y, title):
    for k in range(4):
        cloud = X[y == k]
        col_x = cloud[:, 0]
        col_y = cloud[:, 1]
        ax.scatter(col_x, col_y, s=14, alpha=0.65, color=COLORS[k], label=f"Class {k}")

    means_x = MEANS[:, 0]
    means_y = MEANS[:, 1]
    ax.scatter(means_x, means_y, marker="X", s=160, c="black", zorder=5, label="Class center")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

def nearest_center(points):
    distances = []

    for mean in MEANS:
        distances.append(np.linalg.norm(points - mean, axis=1))

    distances = np.array(distances)
    return distances.argmin(axis=0)

def mixing_rate(points, labels):
    return np.mean(nearest_center(points) != labels)

def separation_ratios():
    sigma = STDS.mean(axis=1)
    ratios = []

    for i in range(4):
        for j in range(i + 1, 4):
            distance = np.linalg.norm(MEANS[i] - MEANS[j])
            ratios.append((i, j, distance / (sigma[i] + sigma[j])))

    return ratios

def main():
    rng = generate_rng()

    # A
    print("\nA. Generate the clouds\n")

    X, y = generate_points(rng, 1.0)
    print(f"{len(X)} samples, 4 classes of 100")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_distributions(ax, X, y, "Figure 1. Gaussian generated distributions with class centers")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig1.png")

    # B1
    print("\nB. More or less spread out\n")

    clouds = {}
    rates = {}

    for scale in SCALES:
        points, labels = generate_points(rng, scale)
        clouds[scale] = points
        rates[scale] = mixing_rate(points, labels)

    all_points = np.vstack(list(clouds.values()))
    xlim = (all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ylim = (all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")

    for ax, scale in zip(axes.ravel(), SCALES):
        plot_distributions(ax, clouds[scale], y, f"scale = {scale}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 2. Same classes at four spread scales")
    save_fig(fig, "fig2.png")

    # B2
    print("\nSeparation ratio (scale = 1)\n")

    ratios = separation_ratios()
    smallest_i, smallest_j, smallest_r = ratios[0]

    for i, j, ratio in ratios:
        print(f"| ({i}, {j}) | r = {ratio:.4f} |")

        if ratio < smallest_r:
            smallest_i, smallest_j, smallest_r = i, j, ratio

    print(f"\nSmallest: pair ({smallest_i}, {smallest_j}) with ratio = {smallest_r:.4f}")
    print(f"At scale = 2, ratio = {smallest_r / 2:.4f}")

    print("\nMixing rate per scale\n")

    for scale in SCALES:
        print(f"scale = {scale}: {rates[scale]:.4f}  ({rates[scale] * 100:.2f}%)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(SCALES, [rates[scale] for scale in SCALES], "o-", color=COLORS[0])

    for scale in SCALES:
        ax.annotate(f"{rates[scale]:.3f}", (scale, rates[scale]), textcoords="offset points", xytext=(0, 9), ha="center")

    ax.set_title("Figure 3. Mixing rate against spread scale")
    ax.set_xlabel("Spread scale")
    ax.set_ylabel("Mixing rate")
    ax.set_xticks(SCALES)
    ax.grid(alpha=0.3)
    save_fig(fig, "fig3.png")

    # C
    print("\nC. Analysis\n")

    print(f"mixing rate at scale = 1: {rates[1.0]:.4f}")
    print(f"mixing rate at scale = 4: {rates[4.0]:.4f}")

if __name__ == "__main__":
    main()
