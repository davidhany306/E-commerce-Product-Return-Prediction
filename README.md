🛒 E-commerce Product Return Prediction

This project predicts whether a product purchased online will be returned or kept by the customer using machine learning techniques.

📘 Overview

The goal is to help e-commerce businesses reduce costs and improve logistics by predicting product returns based on customer and product features.

This project includes:

Data cleaning and preprocessing

Handling categorical and numerical variables

Balancing classes using SMOTE

Model comparison using F1-score

Final Stacking Classifier built from Random Forest and LightGBM

⚙️ Tools & Libraries

Python

Pandas, NumPy

Scikit-learn

LightGBM

XGBoost

Imbalanced-learn

Joblib

🧠 Models Used
Model	Role
Logistic Regression	Meta model
Random Forest	Base learner
LightGBM	Base learner
SMOTE	Balancing technique
Stacking Classifier	Final ensemble model
📂 Files
File	Description
ocelot.pkl	Trained ML model
README.md	Project documentation
🚀 Usage

Clone the repository:

git clone https://github.com/davidhany306/E-commerce-Product-Return-Prediction.git
cd E-commerce-Product-Return-Prediction


Install dependencies:

pip install -r requirements.txt


Load and use the model:

import joblib
model = joblib.load("ocelot.pkl")
predictions = model.predict(X_test)

📈 Results

The stacking model achieved the best F1-score, effectively handling class imbalance and improving return prediction accuracy.

👤 Author
David Hany
🔗 GitHub Profile
