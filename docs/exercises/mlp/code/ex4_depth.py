import matplotlib.pyplot as plt
import numpy as np
from base import COLORS, generate_rng, save_fig
from ex3_multiclass import CLASSES, ETA, EPOCHS, generate_data
from mlp import MLP, accuracy, categorical_cross_entropy, one_hot, softmax

SHALLOW = [4, 16, 3]
DEEP = [4, 16, 16, 3]
SEEDS = [42, 43, 44]

def train_once(sizes, seed, X_train, Y_train):
    model = MLP(sizes, generate_rng(seed), ETA, softmax, categorical_cross_entropy)
    model.train(X_train, Y_train, EPOCHS)

    return model

def run_seeds(sizes, X_train, Y_train, X_test, y_test):
    accuracies = []
    final_losses = []

    for seed in SEEDS:
        model = train_once(sizes, seed, X_train, Y_train)
        accuracies.append(accuracy(model, X_test, y_test))
        final_losses.append(model.losses[-1])
        print(f"  seed {seed}: test accuracy {accuracies[-1]:.4f}, final loss {final_losses[-1]:.4f}")

    return np.mean(accuracies), np.mean(final_losses)

def main():
    X_train, X_test, y_train, y_test = generate_data()
    Y_train = one_hot(y_train, CLASSES)

    # A
    print("\nA. Train the deeper network\n")

    deep = train_once(DEEP, 42, X_train, Y_train)
    print(f"architecture {DEEP}, eta {ETA}, epochs {EPOCHS}, parameters {deep.count_parameters()}")
    print(f"final training loss: {deep.losses[-1]:.4f}")
    print(f"test accuracy: {accuracy(deep, X_test, y_test):.4f}")

    # B
    print("\nB. Compare\n")

    shallow = train_once(SHALLOW, 42, X_train, Y_train)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(shallow.losses, color=COLORS[0], label=f"1 hidden layer {SHALLOW}")
    ax.plot(deep.losses, color=COLORS[1], label=f"2 hidden layers {DEEP}")
    ax.set_title("Figure 6. Training loss, one hidden layer against two")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, "fig6.png")

    print(f"one hidden layer {SHALLOW}, parameters {shallow.count_parameters()}")
    shallow_accuracy, shallow_loss = run_seeds(SHALLOW, X_train, Y_train, X_test, y_test)

    print(f"two hidden layers {DEEP}, parameters {deep.count_parameters()}")
    deep_accuracy, deep_loss = run_seeds(DEEP, X_train, Y_train, X_test, y_test)

    print(f"\nmean over {len(SEEDS)} seeds")
    print(f"  1 layer:  loss {shallow_loss:.4f}, accuracy {shallow_accuracy:.4f}")
    print(f"  2 layers: loss {deep_loss:.4f}, accuracy {deep_accuracy:.4f}")
    print(f"  accuracy difference: {deep_accuracy - shallow_accuracy:+.4f}")
    print(f"  loss difference: {deep_loss - shallow_loss:+.4f}")

if __name__ == "__main__":
    main()
