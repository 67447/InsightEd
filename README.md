# 🎓 InsightEd: School Dropout Prediction & Prevention

**InsightEd** is an AI-powered system designed to predict and prevent school dropouts. It combines machine learning modeling and an interactive Streamlit interface to help educators identify high-risk students and apply targeted interventions.

---

## 📁 Project Structure

This repository contains all the necessary files to run the system locally.

### 1. 📊 Model Development

- **File:** `Data Science Project FIN.ipynb`  
  This Jupyter Notebook contains the full pipeline for:
  - Data cleaning and preprocessing  
  - Feature engineering  
  - Model training using Random Forest  
  - Saving the trained model for use in the web interface  

- **Required Data:**
  - `student-mat.csv`
  - `student-por.csv`

> 📌 Make sure these files are in the same directory as the notebook before running.

---

### 2. 🖥️ Web Interface with Streamlit

- **File:** `streamlit_FIN.py`  
  After training your model in the notebook, launch the interactive web app:

  ```bash
  streamlit run streamlit_FIN.py
