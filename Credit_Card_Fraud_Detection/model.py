# model.py (Task 5 - Credit Card Fraud Detection)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -------------------------
# 1) Load Dataset
# -------------------------
DATA_PATH = "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Credit_Card_Fraud_Detection/creditcard.csv"
SAVE_DIR = "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Credit_Card_Fraud_Detection"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df["Class"].value_counts())

# -------------------------
# 2) Features & Target
# -------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------
# 3) Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4) Preprocessing: Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 5) Handle Class Imbalance with SMOTE
# -------------------------
print("\nBefore SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", np.bincount(y_train_res))

# -------------------------
# 6) Train Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

best_model = None
best_f1 = -np.inf

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n🔹 {name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    
    # Track best model (based on F1-score for fraud class)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report["1"]