# visualization.py (Task 1 - Titanic Survival Prediction)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1) Load Dataset
# -------------------------
df = pd.read_csv(
    r"C:\Users\SAURAV THAKUR\Desktop\Codsoft\Titanic_Survival_Prediction\data\titanic.csv"
)

# -------------------------
# 2) Graphs and Plots
# -------------------------

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="coolwarm")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# Age Distribution
plt.figure(figsize=(8,6))
sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Survival by Age (Boxplot)
plt.figure(figsize=(8,6))
sns.boxplot(x="Survived", y="Age", data=df, palette="pastel")
plt.title("Survival by Age")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# Correlation Heatmap (Numeric Features)
plt.figure(figsize=(8,6))
corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
