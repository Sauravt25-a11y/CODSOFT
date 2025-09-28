# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1) Load Dataset
# -------------------------
df = pd.read_csv(
    "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Movie_Rating_Prediction/IMDb_Movies_India.csv",
    encoding="latin1"
)

# Keep only rows with ratings
df = df.dropna(subset=["Rating"]).copy()

# Clean numeric columns
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# -------------------------
# 2) Graphs and Plots
# -------------------------

# Histogram of IMDb Ratings
plt.figure(figsize=(8,6))
sns.histplot(df["Rating"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of IMDb Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Movie Duration Distribution
plt.figure(figsize=(8,6))
sns.histplot(df["Duration"].dropna(), bins=30, kde=True, color="orange")
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Votes vs Ratings
plt.figure(figsize=(8,6))
sns.scatterplot(x="Votes", y="Rating", data=df, alpha=0.5, color="purple")
plt.title("Votes vs Ratings")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.xscale("log")  # because votes are very skewed
plt.show()

# Boxplot of Ratings by Genre (top 10 genres only)
top_genres = df["Genre"].value_counts().nlargest(10).index
plt.figure(figsize=(12,6))
sns.boxplot(x="Genre", y="Rating", data=df[df["Genre"].isin(top_genres)])
plt.xticks(rotation=45)
plt.title("Ratings by Genre (Top 10 Genres)")
plt.show()

# Correlation Heatmap for Numeric Features
numeric_cols = ["Year", "Duration", "Votes", "Rating"]
plt.figure(figsize=(8,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()
