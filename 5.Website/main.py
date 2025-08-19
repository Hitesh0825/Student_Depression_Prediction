import joblib
import pandas as pd
import streamlit as st
import importlib.util
import sys
import subprocess

# --- Ensure scikit-learn is installed (to load pipeline.pkl safely)
if importlib.util.find_spec("sklearn") is None:
    st.warning("Installing missing package: scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

st.set_page_config(page_title="Depression Prediction App")

st.title("🧠 Depression Prediction App")
st.markdown("""
Welcome to the Depression Prediction System!  
Please fill in the details, and the model will predict whether depression is likely.  
---
""")

# --- Load objects safely ---
try:
    df = joblib.load("df.pkl")
except FileNotFoundError:
    st.error("`df.pkl` not found! Please make sure it's in the project folder.")
    st.stop()

try:
    pipeline = joblib.load("pipeline.pkl")
except FileNotFoundError:
    st.error("`pipeline.pkl` not found! Please make sure it's in the project folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading pipeline.pkl: {e}")
    st.info("This might be due to a version mismatch of scikit-learn. "
            "Try installing the original version used to train the model.")
    st.stop()

st.header("Enter your inputs")

Gender = st.selectbox('Gender', sorted(df['Gender'].unique().tolist()))
Academic_Pressure = st.selectbox('Academic Pressure', sorted(df['Academic Pressure'].unique().tolist()))
Study_Satisfaction = st.selectbox('Study Satisfaction', sorted(df['Study Satisfaction'].unique().tolist()))
Sleep_Duration = st.number_input('Sleep Duration (hours)', min_value=0, max_value=24, step=1)
Dietary_Habits = st.selectbox('Dietary Habits', sorted(df['Dietary Habits'].unique().tolist()))
Degree = st.selectbox('Degree', sorted(df['Degree'].unique().tolist()))
Study_Hours = st.number_input('Study Hours (per day)', min_value=0.0, max_value=24.0, step=0.5)
Financial_Stress = st.selectbox('Financial Stress', sorted(df['Financial Stress'].unique().tolist()))
Age_Category = st.selectbox('Age Category', sorted(df['Age Category'].unique().tolist()))

if st.button('Predict'):
    data = [[Gender, Academic_Pressure, Study_Satisfaction, Sleep_Duration,
             Dietary_Habits, Degree, Study_Hours, Financial_Stress, Age_Category]]

    columns = ['Gender', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration',
               'Dietary Habits', 'Degree', 'Study Hours', 'Financial Stress', 'Age Category']

    one_df = pd.DataFrame(data, columns=columns)

    st.subheader("Your Input:")
    st.dataframe(one_df)

    try:
        prediction = pipeline.predict(one_df)[0]
        if prediction == 1:
            st.error("⚠️ Depression Risk Present")
        else:
            st.success("✅ No Depression Risk")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("This may be due to incompatible sklearn version. Consider retraining or using skops to save models.")

    st.markdown("---")
    st.markdown("Made with ❤️ by Hitesh Sharma")
