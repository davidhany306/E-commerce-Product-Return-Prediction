# E-commerce-Product-Return-Prediction
Predicting product return likelihood in e-commerce with machine learning to support business decision-making.

🛒 E-commerce Sales & Returns Dashboard
This is an interactive web application built with Streamlit that analyzes a comprehensive e-commerce sales dataset. The dashboard provides key performance indicators (KPIs), detailed exploratory data analysis through interactive charts, and a machine learning model to predict whether a customer will return an order.

This project was developed from a detailed analysis initially performed in a Jupyter Notebook.

✨ Features
🏠 Home Page: A high-level overview of the business with key metrics like Total Sales, Average Return Rate, and Average Profit Margin, plus a preview of the raw data.

📊 Interactive EDA Page: A deep dive into the data organized into clean, symmetrical tabs for a superior user experience:

Sales & Profit Analysis

Customer Insights

Regional & Delivery Performance

Returns Analysis

🔮 Return Prediction Tool: Utilizes a trained RandomForest Classifier to predict the likelihood of an order being returned based on user-input features. The results are displayed in a clean, centered layout.

🎨 Modern UI: A custom-designed, dark-themed interface with a professional CSS gradient background that provides a polished and visually appealing experience.

Dynamic Filtering: Users can filter the entire dashboard's data by product category and region in real-time using the interactive sidebar.

🛠️ Technology Stack
Language: Python

Web Framework: Streamlit

Data Manipulation: Pandas

Data Visualization: Plotly Express

Machine Learning: Scikit-learn

Model Explainability: Matplotlib (for styled prediction explanations)

⚙️ Setup and Installation
To run this application on your local machine, please follow these steps:

Clone the Repository

git clone [https://github.com/your-username/ecommerce-dashboard.git](https://github.com/your-username/ecommerce-dashboard.git)
cd ecommerce-dashboard

Create a Virtual Environment (Recommended)

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Required Packages
Create a file named requirements.txt and add the following libraries to it:

streamlit
pandas
plotly
scikit-learn
matplotlib

Then, install them all using pip:

pip install -r requirements.txt

Run the Streamlit App
Make sure you have the ecommerce_sales_34500.csv dataset in the same directory as your script. Then, run the following command in your terminal:

streamlit run final_app.py

The application will automatically open in your default web browser.

📁 File Structure
.
├── final_app.py                # The main Streamlit application script
├── ecommerce_sales_34500.csv     # The dataset for the dashboard
├── final_project.ipynb           # The original Jupyter Notebook with the full analysis
├── requirements.txt              # List of Python dependencies for easy installation
└── README.md                     # You are here!

📊 Dataset
The application uses the ecommerce_sales_34500.csv dataset. This dataset contains detailed, anonymized records of e-commerce transactions, including order details, customer demographics, shipping information, and return status. The initial data cleaning and feature engineering steps are documented in the final_project.ipynb notebook.
