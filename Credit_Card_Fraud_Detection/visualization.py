# visualization.py (Task 5 - Credit Card Fraud Detection)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# 1) Load Dataset
# -------------------------
df = pd.read_csv(
    "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Credit_Card_Fraud_Detection/creditcard.csv"
)

print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df["Class"].value_counts())

# -------------------------
# 2) Save Plots to 'plots' folder inside Credit_Card_Fraud_Detection
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where script is located
PLOTS_DIR = os.path.join(BASE_DIR, "plots")           # plots folder inside same dir
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# 3) Graphs and Plots
# -------------------------

# Fraud vs Genuine Transactions
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df, palette="Set2")
plt.title("Fraud (1) vs Genuine (0) Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.savefig(os.path.join(PLOTS_DIR, "fraud_vs_genuine.png"))
plt.close()

# Distribution of Transaction Amounts
plt.figure(figsize=(8,6))
sns.histplot(df["Amount"], bins=50, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig(os.path.join(PLOTS_DIR, "amount_distribution.png"))
plt.close()

# Boxplot of Transaction Amounts by Class
plt.figure(figsize=(8,6))
sns.boxplot(x="Class", y="Amount", data=df, palette="coolwarm")
plt.title("Transaction Amounts by Class (Fraud vs Genuine)")
plt.xlabel("Transaction Type")
plt.ylabel("Amount")
plt.savefig(os.path.join(PLOTS_DIR, "amount_boxplot.png"))
plt.close()

# Correlation Heatmap (first 10 features + Class for visibility)
plt.figure(figsize=(10,8))
corr = df.iloc[:, :10].join(df["Class"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (first 10 features + Class)")
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.close()

# Scatter Plot (Time vs Amount, colored by Class)
plt.figure(figsize=(10,6))
sns.scatterplot(
    x="Time", y="Amount", hue="Class", data=df, alpha=0.5, palette={0: "green", 1: "red"}
)
plt.title("Transaction Time vs Amount (Fraud vs Genuine)")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.legend(["Genuine", "Fraud"])
plt.savefig(os.path.join(PLOTS_DIR, "time_vs_amount.png"))
plt.close()

print(f"✅ All plots saved inside: {PLOTS_DIR}")
# ---------------------------------------------- 