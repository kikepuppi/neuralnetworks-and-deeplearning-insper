import matplotlib.pyplot as plt
import numpy as np
from base import COLORS, generate_points, generate_rng, plot_boundary, plot_classes, plot_errors, save_fig
from perceptron import Perceptron, initial_weights

MEAN_0 = [3.0, 3.0]
MEAN_1 = [4.0, 4.0]
COV = [[1.5, 0.0], [0.0, 1.5]]
CENTRE = np.array([3.5, 3.5])

def main():
    rng = generate_rng()

    # A
    print("\nA. Generate the data\n")

    X, y = generate_points(rng, MEAN_0, MEAN_1, COV)
    print(f"{len(X)} samples, 1000 per class")
    print(f"mean norm(x) = {np.mean(np.linalg.norm(X, axis=1)):.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_classes(ax, X, y)
    ax.set_title("Figure 4. Overlapping classes")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig4.png")

    # B
    print("\nB. Train, keeping the best weights\n")

    w0, b0 = initial_weights(rng)
    model = Perceptron(w0, b0, 0.01)
    model.train(X, y)

    print(f"final:  w = [{model.w[0]:.4f}, {model.w[1]:.4f}], b = {model.b:.4f}, accuracy = {model.history[-1]:.4f}")
    print(f"pocket: w = [{model.pocket_w[0]:.4f}, {model.pocket_w[1]:.4f}], b = {model.pocket_b:.4f}, accuracy = {model.pocket_accuracy:.4f}")
    print(f"pocket best found at epoch {model.pocket_epoch}, training ran {model.epochs} epochs")

    print(f"final at the cloud centre:  w . [3.5, 3.5] + b = {np.dot(model.w, CENTRE) + model.b:.4f}")
    print(f"pocket at the cloud centre: w . [3.5, 3.5] + b = {np.dot(model.pocket_w, CENTRE) + model.pocket_b:.4f}")

    # C
    print("\nC. Figures\n")

    wrong_final = model.wrong(X, y)
    wrong_pocket = model.wrong_pocket(X, y)
    print(f"misclassified by the final weights: {wrong_final.sum()}")
    print(f"misclassified by the pocket weights: {wrong_pocket.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")

    plot_classes(axes[0], X, y)
    plot_boundary(axes[0], model.w, model.b, X, "Final boundary")
    plot_errors(axes[0], X, wrong_final)
    axes[0].set_title("Final weights")
    axes[0].legend(loc="upper left", fontsize=8)

    plot_classes(axes[1], X, y)
    plot_boundary(axes[1], model.pocket_w, model.pocket_b, X, "Pocket boundary")
    plot_errors(axes[1], X, wrong_pocket)
    axes[1].set_title("Pocket weights")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 5. Final and pocket boundaries on overlapping data")
    save_fig(fig, "fig5.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, model.epochs + 1)
    ax.plot(epochs, model.history, color=COLORS[0], linewidth=1, label="Current weights")
    ax.plot(epochs, model.pocket_history, color=COLORS[1], linewidth=2, label="Pocket (best so far)")
    ax.set_title("Figure 6. Accuracy per epoch, current against pocket")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    save_fig(fig, "fig6.png")

    # D
    print("\nD. Analysis\n")

    print(f"b moves {0.01:.4f} per mistake, w moves about {0.01 * np.mean(np.linalg.norm(X, axis=1)):.4f}")
    print(f"accuracy of the current weights: min {min(model.history):.4f}, max {max(model.history):.4f}")

if __name__ == "__main__":
    main()
