import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="📺"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\KAVIYA V\Downloads\youtube_cleaned_dataset.xls")

df = load_data()

# --------------------------------------------------
# Load Model, Scaler & Feature Columns
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
@st.cache_data
def get_feature_importance(_model, df):
    X = df.select_dtypes(include='number').drop('ad_revenue_usd', axis=1)
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance": _model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Revenue Prediction", "EDA Dashboard", "Model Insights"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg",
        width=200
    )

    st.title("Content Monetization Modeler")
    st.subheader("YouTube Ad Revenue Prediction System")

    st.markdown("""
    ### 🎯 Objective
    Predict YouTube ad revenue using Machine Learning.

    ### 🔧 Features
    - Revenue Prediction  
    - Interactive EDA  
    - Feature Importance  
    - Business Insights  

    ### 🛠 Tech Stack
    Python | Pandas | Scikit-learn | Streamlit
    """)

# --------------------------------------------------
# REVENUE PREDICTION (FINAL FIXED VERSION)
# --------------------------------------------------
elif page == "Revenue Prediction":

    st.title("💰 Revenue Prediction")

    with st.form("prediction_form", clear_on_submit=False):

        col1, col2 = st.columns(2)

        with col1:
            views = st.number_input("Views", min_value=0, key="views")
            likes = st.number_input("Likes", min_value=0, key="likes")
            comments = st.number_input("Comments", min_value=0, key="comments")
            watch_time = st.number_input("Watch Time (minutes)", min_value=0.0, key="watch_time")
            video_length = st.number_input("Video Length (minutes)", min_value=0.0, key="video_length")

        with col2:
            subscribers = st.number_input("Subscribers", min_value=0, key="subscribers")

        submit = st.form_submit_button("Predict Revenue")

    if submit:

        engagement_rate = (st.session_state.likes + st.session_state.comments) / (st.session_state.views + 1)
        likes_per_view = st.session_state.likes / (st.session_state.views + 1)
        comments_per_view = st.session_state.comments / (st.session_state.views + 1)
        watch_ratio = st.session_state.watch_time / (st.session_state.video_length + 1)

        user_input = {
            "views": st.session_state.views,
            "likes": st.session_state.likes,
            "comments": st.session_state.comments,
            "watch_time_minutes": st.session_state.watch_time,
            "video_length_minutes": st.session_state.video_length,
            "subscribers": st.session_state.subscribers,
            "engagement_rate": engagement_rate,
            "likes_per_view": likes_per_view,
            "comments_per_view": comments_per_view,
            "watch_time_ratio": watch_ratio
        }

        input_df = pd.DataFrame([user_input])

        # Add missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"Estimated Ad Revenue: ${prediction:.2f}")
# --------------------------------------------------
# EDA DASHBOARD
# --------------------------------------------------
elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Correlation", "Distribution", "Trends"])

    # ---------- Correlation ----------
    with tab1:

        numeric_cols = df.select_dtypes(include='number').columns
        selected = st.multiselect(
            "Select Features",
            numeric_cols,
            default=list(numeric_cols[:8])
        )

        corr = df[selected].corr()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------- Distribution ----------
    with tab2:

        metric = st.selectbox("Select Metric", numeric_cols)

        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(df[metric], kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

    # ---------- Trends ----------
    with tab3:

        trend_type = st.selectbox(
            "Select Trend",
            ["Category", "Device", "Country"]
        )

        if trend_type == "Category":
            cols = [c for c in df.columns if c.startswith("category_")]
        elif trend_type == "Device":
            cols = [c for c in df.columns if c.startswith("device_")]
        else:
            cols = [c for c in df.columns if c.startswith("country_")]

        trend_df = df[cols].mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8,4))
        trend_df.plot(kind="bar", color="teal", ax=ax)
        st.pyplot(fig)

# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
elif page == "Model Insights":

    st.title("🧠 Model Insights")

    st.success("Best Model: Gradient Boosting Regressor")

    fi_df = get_feature_importance(model, df)

    st.subheader("Top 10 Features Influencing Ad Revenue")

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(
        data=fi_df.head(10),
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("""
    ### Key Findings
    - Views & Watch Time are strongest drivers  
    - Engagement rate boosts earnings  
    - Subscribers improve stability  
    - Geography & device affect CPM  

    ### Business Impact
    - Better content planning  
    - Revenue forecasting  
    - Creator growth strategy  
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Content Monetization Modeler – Streamlit App")
