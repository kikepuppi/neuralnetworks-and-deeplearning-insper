import matplotlib.pyplot as plt
import numpy as np
from base import COLORS, generate_points, generate_rng, plot_boundary, plot_classes, plot_errors, save_fig
from perceptron import Perceptron, initial_weights

MEAN_0 = [1.5, 1.5]
MEAN_1 = [5.0, 5.0]
COV = [[0.5, 0.0], [0.0, 0.5]]

def report(model, name):
    print(f"{name}: w = [{model.w[0]:.4f}, {model.w[1]:.4f}], b = {model.b:.4f}")
    print(f"{name}: epochs = {model.epochs}, accuracy = {model.history[-1]:.4f}")
    print(f"{name}: direction = [{model.direction()[0]:.4f}, {model.direction()[1]:.4f}]")
    print(f"{name}: offset b / norm(w) = {model.b / np.linalg.norm(model.w):.4f}")

def main():
    rng = generate_rng()

    # A
    print("\nA. Generate the data\n")

    X, y = generate_points(rng, MEAN_0, MEAN_1, COV)
    print(f"{len(X)} samples, 1000 per class")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_classes(ax, X, y)
    ax.set_title("Figure 1. Separable classes")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig1.png")

    # B
    print("\nB. Implement the perceptron\n")

    w0, b0 = initial_weights(rng)
    print(f"initial w = [{w0[0]:.6f}, {w0[1]:.6f}], b = {b0}")
    print(f"norm of the initial w = {np.linalg.norm(w0):.6f}")
    print(f"mean norm(x) = {np.mean(np.linalg.norm(X, axis=1)):.4f}")

    # C
    print("\nC. Train and measure\n")

    slow = Perceptron(w0, b0, 0.01)
    slow.train(X, y)
    report(slow, "eta = 0.01")

    wrong = slow.wrong(X, y)
    print(f"eta = 0.01: misclassified = {wrong.sum()}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_classes(ax, X, y)
    plot_boundary(ax, slow.w, slow.b, X, "Decision boundary")
    plot_errors(ax, X, wrong)
    ax.set_title("Figure 2. Decision boundary on separable data")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "fig2.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, slow.epochs + 1)
    ax.plot(epochs, slow.history, "o-", color=COLORS[0], label="Accuracy")
    ax.set_title("Figure 3. Accuracy per epoch, eta = 0.01")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "fig3.png")

    # D
    print("\nD. Analysis\n")

    print("updates per epoch:")

    for epoch, updates in zip(epochs, slow.updates_history):
        print(f"epoch {epoch}: {updates} updates, accuracy {slow.history[epoch - 1]:.4f}")

    fast = Perceptron(w0, b0, 1.0)
    fast.train(X, y)
    report(fast, "eta = 1.00")

    cosine = np.dot(slow.direction(), fast.direction())
    print(f"angle between the two directions = {np.degrees(np.arccos(np.clip(cosine, -1, 1))):.4f} degrees")

    zero_slow = Perceptron(np.zeros(2), 0.0, 0.01)
    zero_slow.train(X, y)

    zero_fast = Perceptron(np.zeros(2), 0.0, 1.0)
    zero_fast.train(X, y)

    print("\nfrom w = 0, b = 0:")
    print(f"eta = 0.01: w = {np.round(zero_slow.w, 6)}, b = {zero_slow.b:.6f}, epochs = {zero_slow.epochs}")
    print(f"eta = 1.00: w = {np.round(zero_fast.w, 6)}, b = {zero_fast.b:.6f}, epochs = {zero_fast.epochs}")
    print(f"ratio w = {np.round(zero_fast.w / zero_slow.w, 6)}, ratio b = {zero_fast.b / zero_slow.b:.6f}")

if __name__ == "__main__":
    main()
