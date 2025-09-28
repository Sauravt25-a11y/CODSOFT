# Movie Rating Prediction System

## Project Overview
This project predicts IMDb movie ratings for Indian movies using a machine learning regression model. It leverages movie-related features like genre, duration, votes, and release year to estimate the movie's expected user rating.

## Features
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA) and Visualization
- Model training with regression algorithms
- Interactive application interface (`app.py`) for making predictions

## Repository Structure
- `data/` - Movie dataset CSV file(s).
- `train_model.py` / `model.py` - Scripts to train and save the regression model.
- `visualization.py` - Script to generate EDA plots and charts.
- `app.py` - Streamlit or Flask app for interactive prediction using the trained model.
- `plots/` - Contains generated visualizations.
- `Requirments.txt` - Required Python dependencies.

## How to Use

### Installation
Install required packages using:
pip install -r Requirments.txt

### Train Model
To train the movie rating prediction model:
python train_model.py

### Run Application
To start the interactive prediction app:
python app.py

Then open the specified local URL in your browser.

## Dataset
- Source: IMDb Movies India dataset (stored in the `data/` folder)
- Includes movie metadata like genre, year, duration, votes, rating, etc.

## Evaluation Metrics
- Root Mean Square Error (RMSE)
- R² Score

## Author
Saurav Thakur

---

This project demonstrates the process from data exploration to interactive machine learning model deployment in Python.
