import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Project folder
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("student_data.csv")

print("\n========== EDA PROJECT ==========")

# 1. Dataset Preview
print("\n1. First 5 Rows:")
print(df.head())

# 2. Dataset Shape
print("\n2. Dataset Shape:")
print(df.shape)

# 3. Dataset Information
print("\n3. Dataset Information:")
df.info()

# 4. Missing Values
print("\n4. Missing Values:")
print(df.isnull().sum())

# 5. Duplicate Values
print("\n5. Duplicate Rows:")
print(df.duplicated().sum())

# 6. Statistical Summary
print("\n6. Statistical Summary:")
print(df.describe())

# 7. Numeric Columns
numeric_columns = df.select_dtypes(include="number").columns

# Histogram
for column in numeric_columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(f"results/{column}_histogram.png")
    plt.close()

# 8. Correlation Heatmap
if len(numeric_columns) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_columns].corr(),
                annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("results/correlation_heatmap.png")
    plt.close()

# 9. Categorical Column Analysis
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    df[column].value_counts().head(10).plot(kind="bar")
    plt.title(f"Count of {column}")
    plt.tight_layout()
    plt.savefig(f"results/{column}_count.png")
    plt.close()

# 10. Save EDA Summary
with open("results/eda_summary.txt", "w") as file:
    file.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
    file.write(str(df.describe()))
    file.write("\n\nMissing Values:\n")
    file.write(str(df.isnull().sum()))
    file.write("\n\nDuplicate Rows:\n")
    file.write(str(df.duplicated().sum()))

print("\nEDA ANALYSIS COMPLETED!")
print("All graphs saved in results folder.")