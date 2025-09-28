# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------ Load Model ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at {path}. Please run train_model.py first.")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ------------------ Title ------------------ #
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster based on their details.")

# ------------------ Sidebar Inputs ------------------ #
st.sidebar.header("Passenger Details")

def user_input_features():
    Pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.sidebar.radio("Sex", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 0, 80, 30)
    SibSp = st.sidebar.slider("Siblings/Spouses aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Parents/Children aboard", 0, 6, 0)
    Fare = st.sidebar.slider("Fare ($)", 0.0, 500.0, 32.0)
    Embarked = st.sidebar.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])
    
    # Encode categorical
    Sex_encoded = 1 if Sex == "Male" else 0
    Embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    Embarked_encoded = Embarked_map[Embarked]
    
    data = {
        "Pclass": Pclass,
        "Sex": Sex_encoded,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked_encoded
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ------------------ Prediction ------------------ #
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.image(
    "https://th.bing.com/th/id/R.036e32952ebb3e5d64797f0a8d125cc8?rik=zXe8bmLIxQ%2bgfg&riu=http%3a%2f%2ffilm.org.pl%2fwp-content%2fuploads%2f2012%2f04%2fShips_Titanic_014255_1.jpg&ehk=9B%2funlFHYBvgL0FGGdW0lMSCxJe%2bX%2b1IJg%2beZ7%2bh0JM%3d&risl=&pid=ImgRaw&r=0",
    caption="The RMS Titanic (1912)"
)

st.subheader("Prediction")
survival_status = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
st.write(f"**Prediction:** {survival_status}")
st.write(f"**Survival Probability:** {prediction_proba[0][1]*100:.2f}%")

# ------------------ Footer ------------------ #
st.markdown("---")
st.write("👨‍💻 Developed by Saurav Thakur")