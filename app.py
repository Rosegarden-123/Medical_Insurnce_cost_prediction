import pickle
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import pickle
import numpy as np
import pandas as pd
with open(r"C:\Users\Pitchamani\Desktop\Insurance_cost_prediction\env\Scripts\cost_prediction\gradient_boost_model2.pkl", "rb") as f:
    model = pickle.load(f)

# Set the page config
st.set_page_config(page_title='Data Visualizer',
                   layout='centered',
                   page_icon='📊')
st.image(r"C:\Users\Pitchamani\Desktop\Insurance_cost_prediction\env\Scripts\cost_prediction\image.png",caption="g",use_container_width=True)
st.sidebar.title("Medical Insurance Predictor")
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0)
children = st.sidebar.number_input("Number of children", min_value=0, max_value=10)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_df = pd.DataFrame([{
    "age": age,
    "sex": 0 if sex == "male" else 1,
    "bmi": bmi,
    "children": children,
    "smoker": 0 if smoker == "yes" else 1,
    "region": {"southeast":0,"southwest":1,"northwest":2,"northeast":3}[region]
}])

if st.sidebar.button("Predict Insurance Cost"):
    pred = model.predict(input_df)
    st.sidebar.success(f"Estimated Cost: ${pred[0]:,.2f}")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df_insurance = pd.read_csv(r"C:\Users\Pitchamani\Desktop\Medical Insurance cost prediction\cleaned.csv")

# Sidebar or dropdown menu
st.title("EDA Menu")

eda_option = st.selectbox(
    "Select EDA Analysis",
    (
        "Distribution of Medical Insurance Charges",
        "Age distribution of the individuals",
        "Distribution of BMI",
        "Charges vs Age Scatterplot",
        "Average Charges by Region"
    )
)

# EDA Plots
if eda_option == "Distribution of Medical Insurance Charges":
    st.subheader("Distribution of Insurance Charges")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(df_insurance['charges'], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title('Distribution of Insurance Charges')
    ax.set_xlabel('Charges')
    ax.set_ylabel('Number of Insurers')
    st.pyplot(fig)

if eda_option == "Age distribution of the individuals":
    st.subheader("Age Distribution of Individuals")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(df_insurance['age'], bins=20, kde=True, color='lightblue', ax=ax)
    ax.set_title('Age Distribution of Individuals')
    ax.set_xlabel('age')
    ax.set_ylabel('count')
    st.pyplot(fig)

elif eda_option == "Distribution of BMI":
    st.subheader("Distribution of BMI")
    fig, ax = plt.subplots()
    sns.histplot(df_insurance['bmi'], kde=True, color='orange', ax=ax)
    st.pyplot(fig)

elif eda_option == "Charges vs Age Scatterplot":
    st.subheader("Charges vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_insurance, x='age', y='charges', hue='smoker', ax=ax)
    st.pyplot(fig)

elif eda_option == "Average Charges by Region":
    st.subheader("Average Charges by Region")
    avg_charges = df_insurance.groupby("region")["charges"].mean().sort_values()
    fig, ax = plt.subplots()
    avg_charges.plot(kind='barh', color='teal', ax=ax)
    ax.set_xlabel("Average Charges")
    st.pyplot(fig)

