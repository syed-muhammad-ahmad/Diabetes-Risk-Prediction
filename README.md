# 🏥 Diabetes Readmission Risk Prediction

A Machine Learning project to predict whether a diabetic patient will be readmitted to the hospital within 30 days. The project includes data analysis, model training, and a live Streamlit dashboard for real-time predictions.

## 📌 Problem Statement
Hospital readmissions are costly and indicate poor patient outcomes.  
**Goal:** Use patient data to predict readmission risk early so hospitals can provide targeted preventive care and reduce costs.

## 📊 Dataset
- **Source:** UCI Machine Learning Repository - Diabetes 130-US hospitals dataset
- **Size:** ~100,000 patient records
- **Target:** `readmitted` - No, >30, <30

## 🛠️ Methodology

1.  **Data Cleaning**  
    - Handled missing values marked as '?'
    - Dropped columns with >40% missing data
    - Encoded categorical variables

2.  **Feature Engineering**  
    - `Total_Medications`: Sum of all diabetes medications
    - `Long_Stay`: Flag for hospital stay > 7 days

3.  **Handling Imbalance**  
    - Used `SMOTE` from imbalanced-learn to balance classes

4.  **Model Training & Tuning**  
    - Models: Logistic Regression, Random Forest, XGBoost
    - Tuning: GridSearchCV with ROC-AUC scoring
    - **Best Model: XGBoost** with ROC-AUC ~ 0.67

5.  **Deployment**  
    - Built an interactive web app using `Streamlit`
    - Doctors can input patient details and get `High Risk / Low Risk` prediction with probability

## 🚀 How to Run This Project

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Diabetes-Risk-Prediction.git
cd Diabetes-Risk-Prediction
```
2.install requirements 
```bash
pip install -r requirements.txt
```

3.Run the Streamlit Dashboard 
```bash
streamlit run app.py
```

4.Project structure 
```bash
Diabetes-Risk-Prediction/
│
├── Diabetes_Risk_Analysis.ipynb  # Jupyter notebook with full EDA and training
├── app.py                        # Streamlit dashboard code
├── model.joblib                  # Trained XGBoost model
├── scaler.joblib                 # Fitted scaler
├── columns.joblib                # Training columns for alignment
├── requirements.txt              # Dependencies
└── README.md                     # You are here
```
Key Insight
XGBoost performed best due to its ability to handle non linear relationships and class imbalance 

⭐if you find it useful , consider giving star!
