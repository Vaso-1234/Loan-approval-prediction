# Loan Approval Prediction

## Project Overview
This project predicts whether a loan application is approved or rejected based on applicant details:
- Age
- Income
- Loan Amount
- Credit Score

The project is developed using **Python, Streamlit, and Machine Learning**.

---

## Features
- Simple and attractive Streamlit interface
- Loan approval / rejection prediction
- Input validation for invalid values
- Separate result page for prediction output
- Rejection reason display for ineligible applicants

---

## Files Included
- `app.py` → Main Streamlit web app
- `train_model.py` → Model training script
- `preprocess.py` → Data preprocessing script
- `loan_data.csv` → Dataset
- `model.pkl` → Trained model
- `scaler.pkl` → Saved scaler
- `evaluation_report.json` → Model evaluation report
- `requirements.txt` → Required Python libraries

---

## How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt