import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "loan_data.csv"

FEATURES = ["Age", "Income", "LoanAmount", "CreditScore"]
TARGET = "Approved"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    df = df.copy()

    print("\nBefore preprocessing:", df.shape)
    df.drop_duplicates(inplace=True)

    df = df[df["Age"] >= 18]
    df = df[df["Income"] > 0]
    df = df[df["LoanAmount"] > 0]
    df = df[df["CreditScore"] > 0]

    print("After preprocessing:", df.shape)
    return df


def split_data(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test