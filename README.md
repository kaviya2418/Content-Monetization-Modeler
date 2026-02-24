📺 Content Monetization Modeler
YouTube Ad Revenue Prediction Using Machine Learning
📌 Project Overview

The Content Monetization Modeler is a machine learning project designed to predict YouTube ad revenue based on video performance, engagement metrics, and contextual information.
The project helps content creators and media companies make data-driven decisions for content strategy and revenue planning.

🎯 Problem Statement

As YouTube becomes a primary income source for creators, accurately predicting ad revenue is essential. Revenue depends on multiple factors such as views, engagement, watch time, and audience reach. Manual estimation is unreliable, motivating the need for a predictive model.

🎯 Project Objectives

Predict YouTube ad revenue using regression models

Analyze key factors influencing monetization

Perform exploratory data analysis (EDA)

Build and evaluate multiple regression models

Deploy the solution using a Streamlit web application

📊 Dataset Description

Dataset Name: YouTube Monetization Modeler

Format: CSV

Size: ~122,000 rows

Type: Synthetic dataset (for learning purposes)

Key Features

views, likes, comments

watch_time_minutes, video_length_minutes

subscribers

category, device, country

Target Variable: ad_revenue_usd

🛠 Tools & Technologies

Python – Core programming language

Pandas & NumPy – Data manipulation

Seaborn & Matplotlib – Visualization

Scikit-learn – Machine learning models

Streamlit – Web application deployment

Joblib – Model persistence

🔄 Project Workflow

Data loading and inspection

Data cleaning (missing values & duplicates)

Exploratory Data Analysis (EDA)

Outlier detection using Z-score

Feature engineering

Model training and evaluation

Best model selection

Streamlit app development

🧪 Feature Engineering

Engagement Rate

Likes per View

Comments per View

Watch Time Ratio

These features improve model performance by capturing user engagement behavior.

🤖 Models Used

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Gradient Boosting Regressor (Best Model)

📈 Model Evaluation Metrics

R² Score

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

Gradient Boosting achieved the highest R² score and lowest error.

💻 Streamlit Application Features

Home Page: Project overview and navigation

Revenue Prediction: User input–based revenue estimation

EDA Dashboard:

Correlation analysis

Distribution plots

Trends by category, device, and country

Model Insights: Top 10 features influencing ad revenue

💡 Key Insights

Views and watch time are the strongest drivers of ad revenue

Engagement metrics significantly improve earnings

Subscriber count provides revenue stability

Geography and device type influence CPM

📌 Business Use Cases

Content strategy optimization

Revenue forecasting for creators

Media planning and ad campaign evaluation

Analytics support tools for YouTubers

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn seaborn matplotlib streamlit joblib
2️⃣ Run Streamlit App
streamlit run app.py
🔮 Future Enhancements

Real-time data integration

Advanced models (XGBoost, Deep Learning)

Time-series revenue forecasting

User authentication and dashboards
