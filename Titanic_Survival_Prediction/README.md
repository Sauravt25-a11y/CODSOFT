# 🚢 Titanic Survival Prediction  

This project predicts whether a passenger on the Titanic would survive or not using **Machine Learning** and provides an interactive **Streamlit web app** for simulation.  

---

## ✨ Features  
- Interactive input for passenger details (age, class, sex, family aboard, fare, port of embarkation)  
- Real-time survival prediction using a trained ML model  
- Future upgrades: survival probabilities, EDA visualizations (survival counts, heatmaps, feature importance), and improved UI/UX  
- Simple, user-friendly interface built with **Streamlit**  

---

## ⚙️ Steps to Run  

1. **Clone the repository**  

   git clone https://github.com/Sauravt25-a11y/CODSOFT/tree/main/Titanic_Survival_Prediction
   cd Titanic_Survival_Prediction_Model

2. Install requirements

   pip install -r requirements.txt

3. Run the visualizations (optional)

   python visualizations.py

   This will generate plots.

4. Train the model (if not already trained)

   python train_model.py
   
   This will generate titanic_model.pkl.

5. Run the Streamlit app

   streamlit run app.py

6. Open your browser at:

   http://localhost:8501/


---📂 Project Structure---

Titanic_Survival_Prediction_Model/
│-- app.py                # Streamlit application  
│-- train_model.py        # Script to train and save the ML model
|-- visulization.py       #for generating plots and EDA 
│-- titanic_model.pkl     # Saved ML model (generated after training)  
│-- requirements.txt      # Required Python libraries  
│-- README.md             # Project documentation 
|-- data
   |-- titanic.csv        # dataset
|--plots                  # graphs and EDA pictures


---🛠️ Tech Stack---

   Python
   Pandas, NumPy, Scikit-learn – data preprocessing & ML model
   Streamlit – interactive web app
   Seaborn, Matplotlib – data visualization
