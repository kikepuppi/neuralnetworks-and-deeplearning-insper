import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from base import COLORS, DATA, SEED, save_fig

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["Cabin", "Name", "PassengerId"]
TARGET = "Transported"
NUMERICAL = ["Age"] + SPEND

def describe(df):
    print("\nA. Getting to know the data\n")

    print(f"shape: {df.shape[0]} rows, {df.shape[1]} columns")

    balance = df[TARGET].value_counts(normalize=True).sort_index()

    for label, share in balance.items():
        print(f"Transported = {label}: {share:.4f}  ({share * 100:.2f}%)")

    print(f"\nNumerical: {', '.join(NUMERICAL)}")
    print(f"Categorical: {', '.join(CATEGORICAL)}")

    print("\nMissing values per column\n")

    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "percent": (df.isna().mean() * 100).round(2),
    })
    print(missing[missing["missing"] > 0])

    print("\nSpending columns:\n")

    print(df[SPEND].agg(["mean", "median", "max"]).round(2))

def split(df):
    print("\nB. Split before transforming\n")

    X = df.drop(columns=DROP + [TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    print(f"train: {X_train.shape[0]} rows, Transported = True: {y_train.mean():.4f}")
    print(f"test:  {X_test.shape[0]} rows, Transported = True: {y_test.mean():.4f}")
    print(f"FoodCourt mean on train: {X_train['FoodCourt'].mean():.4f}")

    return X_train, X_test

def impute(X, number_imputer, category_imputer):
    numbers = pd.DataFrame(number_imputer.transform(X[NUMERICAL]), columns=NUMERICAL, index=X.index)
    categories = pd.DataFrame(category_imputer.transform(X[CATEGORICAL]), columns=CATEGORICAL, index=X.index)

    return numbers, categories

def encode(categories, encoder):
    names = encoder.get_feature_names_out(CATEGORICAL)

    return pd.DataFrame(encoder.transform(categories), columns=names, index=categories.index)

def add_total_spend_and_log(numbers):
    numbers = numbers.copy()
    numbers["TotalSpend"] = numbers[SPEND].sum(axis=1)

    for column in SPEND + ["TotalSpend"]:
        numbers[column] = np.log1p(numbers[column])

    return numbers

def scale(numbers, scaler):
    return pd.DataFrame(scaler.transform(numbers), columns=numbers.columns, index=numbers.index)

def preprocess(X_train, X_test):
    print("\nC. Preprocessing\n")

    number_imputer = SimpleImputer(strategy="median").fit(X_train[NUMERICAL])
    category_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[CATEGORICAL])

    for column in NUMERICAL:
        print(f"{column}: median {X_train[column].median()}")

    for column in CATEGORICAL:
        print(f"{column}: mode {X_train[column].mode()[0]}")

    train_numbers, train_categories = impute(X_train, number_imputer, category_imputer)
    test_numbers, test_categories = impute(X_test, number_imputer, category_imputer)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_categories)
    one_hot_names = encoder.get_feature_names_out(CATEGORICAL)
    print(f"\nOneHot columns: {', '.join(one_hot_names)}")

    raw_foodcourt = train_numbers["FoodCourt"].copy()

    train_numbers = add_total_spend_and_log(train_numbers)
    test_numbers = add_total_spend_and_log(test_numbers)

    logged_foodcourt = train_numbers["FoodCourt"].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_numbers)

    scaled_train = scale(train_numbers, scaler)
    scaled_test = scale(test_numbers, scaler)

    print(f"\ntrain numeric range after scaling: [{scaled_train.to_numpy().min():.4f}, {scaled_train.to_numpy().max():.4f}]")
    print(f"test numeric range after scaling:  [{scaled_test.to_numpy().min():.4f}, {scaled_test.to_numpy().max():.4f}]")

    train = pd.concat([scaled_train, encode(train_categories, encoder)], axis=1)
    test = pd.concat([scaled_test, encode(test_categories, encoder)], axis=1)

    return train, test, raw_foodcourt, logged_foodcourt

def plot_log_effect(raw_foodcourt, logged_foodcourt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    axes[0].hist(raw_foodcourt, bins=60, color=COLORS[0])
    axes[0].set_title("FoodCourt, raw amounts")
    axes[0].set_xlabel("FoodCourt (credits)")

    axes[1].hist(logged_foodcourt, bins=60, color=COLORS[2])
    axes[1].set_title("FoodCourt after log(1 + x)")
    axes[1].set_xlabel("log(1 + FoodCourt)")

    for ax in axes:
        ax.set_ylabel("Passengers (log scale)")
        ax.set_yscale("log")

    fig.suptitle("Figure 6. Effect of the log transform")
    save_fig(fig, "fig6.png")

def verify(name, frame):
    values = frame.to_numpy(dtype=float)
    missing = np.isnan(values).sum()

    print(f"{name}: shape {frame.shape}, NaN = {missing}, range [{values.min():.4f}, {values.max():.4f}]")

def main():
    df = pd.read_csv(DATA / "train.csv")

    # A
    describe(df)

    # B
    X_train, X_test = split(df)

    # C
    train, test, raw_foodcourt, logged_foodcourt = preprocess(X_train, X_test)

    # D
    plot_log_effect(raw_foodcourt, logged_foodcourt)

    print("\nD. Final checks\n")

    verify("train", train)
    verify("test", test)

if __name__ == "__main__":
    main()
