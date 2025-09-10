# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------ Load Model ------------------ #
MODEL_PATH = "titanic_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ------------------ Load Model with Caching ------------------ #
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ------------------ Title ------------------ #
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster based on their details.")

# ------------------ Sidebar Inputs ------------------ #
st.sidebar.header("Passenger Details")

def user_input_features():
    Pclass = st.sidebar.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: f"{x} ({'1st' if x==1 else '2nd' if x==2 else '3rd'} Class)"
    )
    Sex = st.sidebar.radio("Sex", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 0, 80, 30)
    SibSp = st.sidebar.slider("Siblings/Spouses aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Parents/Children aboard", 0, 6, 0)
    Fare = st.sidebar.slider("Fare ($)", 0.0, 500.0, 32.0)
    Embarked = st.sidebar.selectbox(
        "Port of Embarkation",
        ["Cherbourg, France", "Queenstown, Ireland", "Southampton, England"]
    )
    
    # Encode categorical inputs for model
    Sex_encoded = 1 if Sex == "Male" else 0
    Embarked_map = {"Cherbourg, France": 0, "Queenstown, Ireland": 1, "Southampton, England": 2}
    Embarked_encoded = Embarked_map[Embarked]
    
    # Create dataframe
    data = {
        "Pclass": Pclass,
        "Sex": Sex_encoded,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked_encoded
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# ------------------ Prediction ------------------ #
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
survival_status = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
st.write(f"**Prediction:** {survival_status}")
st.write(f"**Survival Probability:** {prediction_proba[0][1]*100:.2f}%")

# ------------------ EDA & Insights ------------------ #
st.subheader("Titanic Dataset Insights")

# Load dataset for visualization
csv_path = r"data/titanic.csv"  # adjust path if needed
df = pd.read_csv(csv_path)

# Quick stats
st.write(df.describe())

# Survival count plot
st.write("### Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x="Survived", data=df, ax=ax1)
ax1.set_xticklabels(["Did Not Survive", "Survived"])
st.pyplot(fig1)

# Select numeric columns only
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation heatmap
st.write("### Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)


# Feature importance (Random Forest)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
X = df.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
# Encode categorical
X["Sex"] = X["Sex"].map({"male": 1, "female": 0})
X["Embarked"] = X["Embarked"].map({"C":0, "Q":1, "S":2})
y = df["Survived"]
rf.fit(X, y)

st.write("### Feature Importance (Random Forest)")
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig3, ax3 = plt.subplots()
sns.barplot(x=feat_importances.index, y=feat_importances.values, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)

# ------------------ Footer ------------------ #
st.markdown("---")
st.write("👨‍💻 Developed by Saurav Thakur")
