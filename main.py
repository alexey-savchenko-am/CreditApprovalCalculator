from data_loader import CreditDataset
from pandas import DataFrame, Series
from typing import NoReturn, Tuple
import time
import logistic_regression
import numpy as np

def main() -> NoReturn:
    path: str = "realistic_credit_dataset.csv"
    dataset = CreditDataset(path)
    dataset.load()

    columns_to_check = ["income", "loan_amount", "employment_years", "credit_score"]
    dataset.df = dataset.df[(dataset.df[columns_to_check] >= 0).all(axis=1)]

    print("\n📊 Dataset summary:")
    print(dataset.describe())

    features_series: Tuple[DataFrame, Series] = dataset.get_features_and_target()
    X_train, y_train = features_series

    # Normalize
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train = (X_train - X_mean) / X_std

    model = logistic_regression.LogisticRegressionGD(learning_rate=0.01, n_iter=10000)

    print("\n🔧 Training", end="")
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)

    print("\n")
    model.fit(X_train, y_train)
    print("✅ Training finished.")

    params = model.get_params()
    for i, w in enumerate(params[0]):
        print(f"\nw{i}: {w}")
    print(f"\nbias={params[1]}")

    # 🎯 Interactive loop
    print("\n📥 Enter borrower data to predict approval probability.")
    print("🔚 Type 'exit' to quit.\n")

    feature_names = [
        "age", "income", "loan_amount",
        "employment_years", "has_previous_loans",
        "credit_score", "defaulted_before"
    ]

    while True:
        try:
            user_input = input("Enter values (comma-separated): age,income,loan_amount,employment_years,has_previous_loans,credit_score,defaulted_before\n> ")
            if user_input.strip().lower() == "exit":
                print("👋 Exiting...")
                break

            values = [float(v.strip()) for v in user_input.split(",")]
            if len(values) != 7:
                print("❌ Please enter exactly 7 comma-separated values.")
                continue

            user_data = DataFrame([values], columns=feature_names)
            user_data = (user_data - X_mean) / X_std  # Normalize like training data

            approved = model.predict(user_data)[0]
            result = "yes" if approved == 1 else "no"
            print(f"🧠 Credit approved: {result}")

        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()
