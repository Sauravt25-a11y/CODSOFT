import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import os
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Full path to dataset
csv_path = r"C:\\Users\SAURAV THAKUR\\OneDrive\Desktop\\Codsoft\\Titanic_Survival_Prediction\data\\titanic.csv"

# Check if file exist or not
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

# Load dataset
df = pd.read_csv(csv_path)

# Drop unnecessary columns
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Fill missing values (age with mean, embarked with mode)
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# Encode categorical variables
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform( df["Embarked"])

#--------------------EDA and Visualization--------------------#
# Visualize data Survival distribution
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")

#Random Forest Classifier
plt.figure(figsize=(10, 6))
sns.barplot(x=df.columns[1:], y=RandomForestClassifier().fit(df.drop("Survived", axis=1), df["Survived"]).feature_importances_)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()

# Feature importance with RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(df.drop(["Survived", "PassengerId"], axis=1), df["Survived"])

plt.figure(figsize=(10, 6))
sns.barplot(x=df.drop(["Survived", "PassengerId"], axis=1).columns,
            y=rf.feature_importances_)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=45)
plt.show()

# Define features and target
X = df.drop(["Survived", "PassengerId"], axis=1)  # Features
y = df["Survived"]  # Target

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Ensure the model is saved in the same folder as train_model.py and app.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

#Delete existing model file if exists
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"🗑️ Deleted existing model at {MODEL_PATH}")

# Save model in the same folder as app.py in the titanic survival prediction project
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved as titanic_model.pkl")

print("👨‍💻 Developed by Saurav Thakur")