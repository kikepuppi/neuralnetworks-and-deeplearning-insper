import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from base import generate_rng, plot_loss, save_fig
from mlp import MLP, accuracy, categorical_cross_entropy, one_hot, softmax

CLASSES = 3
SIZES = [4, 16, 3]
ETA = 0.5
EPOCHS = 2000

def generate_data():
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0,
                               random_state=42)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def plot_confusion(ax, matrix):
    ax.imshow(matrix, cmap="Blues")

    for i in range(CLASSES):
        for j in range(CLASSES):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color)

    ax.set_xticks(range(CLASSES), [f"Class {k}" for k in range(CLASSES)])
    ax.set_yticks(range(CLASSES), [f"Class {k}" for k in range(CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

def main():
    rng = generate_rng()

    # A
    print("\nA. Generate the data\n")

    X_train, X_test, y_train, y_test = generate_data()
    print(f"train {X_train.shape}, test {X_test.shape}")

    for k in range(CLASSES):
        print(f"class {k} share on train: {np.mean(y_train == k):.4f}")

    # B
    print("\nB. Softmax and categorical cross-entropy\n")

    Y_train = one_hot(y_train, CLASSES)
    print(f"one-hot targets {Y_train.shape}, each row sums to {Y_train.sum(axis=1)[0]:.1f}")

    # C
    print("\nC. Train and evaluate\n")

    model = MLP(SIZES, rng, ETA, softmax, categorical_cross_entropy)
    model.train(X_train, Y_train, EPOCHS)

    print(f"architecture {SIZES}, eta {ETA}, epochs {EPOCHS}, parameters {model.count_parameters()}")
    print(f"first training loss: {model.losses[0]:.4f}")
    print(f"final training loss: {model.losses[-1]:.4f}")
    print(f"test accuracy: {accuracy(model, X_test, y_test):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_loss(ax, model.losses, "Figure 4. Training loss per epoch")
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, "fig4.png")

    matrix = confusion_matrix(y_test, model.predict(X_test))
    print("\nconfusion matrix, rows are true and columns are predicted:")
    print(matrix)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_confusion(ax, matrix)
    ax.set_title("Figure 5. Confusion matrix on the test set")
    save_fig(fig, "fig5.png")

    # D
    print("\nD. Analysis\n")

    for i in range(CLASSES):
        for j in range(i + 1, CLASSES):
            print(f"pair ({i}, {j}): {matrix[i, j] + matrix[j, i]} errors in both directions")

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"logistic regression test accuracy: {baseline.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    main()
