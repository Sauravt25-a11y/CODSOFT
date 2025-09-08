import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

# ✅ Full path to dataset
csv_path = r"C:\Users\SAURAV THAKUR\OneDrive\Desktop\Codsoft\Titanic_Survival_Prediction\data\titanic.csv"

# Check if file exist or not
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

# Load dataset
df = pd.read_csv(csv_path)

# Drop unnecessary columns
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Fill missing values (age with mean, embarked with mode)
df["Age"].fillna(df["Age"].mean(),  inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform( df["Embarked"])

# Spliting dataset
X = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model in the same folder as app.py in the titanic survival prediction project
joblib.dump(model, "titanic_model.pkl ")
print("✅ Model trained and saved as titanic_model.pkl")

print("👨‍💻 Developed by Saurav Thakur")