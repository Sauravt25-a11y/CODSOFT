🏦 Credit Card Fraud Detection
📌 Overview

This project builds a machine learning system to detect fraudulent credit card transactions.
The dataset is highly imbalanced (very few frauds compared to genuine transactions), so special care was taken to balance it using SMOTE (Synthetic Minority Oversampling Technique).

Two models were trained and compared:

Logistic Regression

Random Forest Classifier

📊 Dataset

File: creditcard.csv

Rows: 284,807

Columns: 31 (Time, Amount, V1…V28, Class)

Target variable:

0 → Genuine Transaction

1 → Fraudulent Transaction

⚙️ Workflow

Data Preprocessing

Scaled numerical features (StandardScaler).

Applied SMOTE to handle class imbalance.

Model Training

Logistic Regression.

Random Forest Classifier.

Evaluation Metrics

Confusion Matrix.

Precision, Recall, and F1-score.

Accuracy.

📈 Results
Logistic Regression

Precision (Fraud): 5.8%

Recall (Fraud): 91.8%

F1-score (Fraud): 10.8%

Very high recall but poor precision (detects frauds but with many false alarms).

Random Forest

Precision (Fraud): 87.1%

Recall (Fraud): 82.6%

F1-score (Fraud): 84.8%

Much better balance → detects most frauds with fewer false alarms.

✅ Random Forest outperformed Logistic Regression.

📊 Visualizations

The following plots are included in visualization.py:

Fraud vs Genuine Transactions Count

Transaction Amount Distribution

Boxplot of Amount by Class

Correlation Heatmap

Time vs Amount Scatterplot (colored by class)

Plots are saved in the plots/ folder.

🚀 How to Run

Clone the repo:

git clone https://github.com/Sauravt25-a11y/CODSOFT.git
cd CODSOFT/Credit_Card_Fraud_Detection


Install dependencies:

pip install -r requirements.txt


Run the model training:

python model.py


Run the visualizations:

python visualization.py

🏆 Conclusion

Random Forest is the best model for this dataset.

Future improvements could include:

Trying XGBoost / LightGBM.

Using cross-validation.

Fine-tuning hyperparameters.