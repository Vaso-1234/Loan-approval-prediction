import pandas as pd

df = pd.read_csv("loan_data.csv")

print("===== SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== APPROVED VALUE COUNTS =====")
print(df["Approved"].value_counts())

print("\n===== CREDIT SCORE STATS =====")
print(df["CreditScore"].describe())

print("\n===== INCOME STATS =====")
print(df["Income"].describe())

print("\n===== LOAN AMOUNT STATS =====")
print(df["LoanAmount"].describe())

print("\n===== LOW CREDIT SCORE CASES (<= 400) =====")
low_credit = df[df["CreditScore"] <= 400]
print(low_credit[["Age", "Income", "LoanAmount", "CreditScore", "Approved"]].head(20))
print("\nCount:")
print(low_credit["Approved"].value_counts())

print("\n===== VERY LOW INCOME CASES (<= 1000) =====")
low_income = df[df["Income"] <= 1000]
print(low_income[["Age", "Income", "LoanAmount", "CreditScore", "Approved"]].head(20))
print("\nCount:")
print(low_income["Approved"].value_counts())

print("\n===== BOTH LOW INCOME + LOW CREDIT =====")
combo = df[(df["Income"] <= 1000) & (df["CreditScore"] <= 400)]
print(combo[["Age", "Income", "LoanAmount", "CreditScore", "Approved"]].head(30))
print("\nCount:")
print(combo["Approved"].value_counts())