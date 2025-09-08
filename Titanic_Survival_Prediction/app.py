import streamlit as st
import joblib
import os

# Absolute path to the model
MODEL_PATH = r"C:\Users\SAURAV THAKUR\OneDrive\Desktop\Codsoft\Titanic_Survival_Prediction\titanic_model.pkl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    st.stop()

# Load trained model
model = joblib.load(MODEL_PATH)

# Streamlit App
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction Model")
st.write("Enter passenger details below to predict survival, Let's see who will survive and break the matrix:")

# User inputs
pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare ($)", 0, 600, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# Prediction
if st.button("Predict Survival"):
    data = [[pclass, sex, age, sibsp, parch, fare, embarked]]
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("🎉 The passenger would have SURVIVED!")
    else:
        st.error("☠️ The passenger would NOT have survived.")

st.write("👨‍💻 Developed by Saurav Thakur")
st.write("📂 [GitHub Repository]( https://github.com/Sauravt25-a11y/Titanic_Survival_Prediction_Model )")