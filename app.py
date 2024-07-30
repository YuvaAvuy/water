import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
data = pd.read_csv("cleaned_ad_data.csv")

# Ensure 'Clicked on Ad' is in binary format
data["Clicked on Ad"] = data["Clicked on Ad"].map({"No": 0, "Yes": 1})

# Prepare data for machine learning model
features = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Gender"]
x = data[features]
y = data["Clicked on Ad"]

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=4)

# Train the Random Forest model
model = RandomForestClassifier(random_state=4)
model.fit(xtrain, ytrain)

# Define the Streamlit app
def main():
    st.title('Ads Click Through Rate Prediction')

    # Collect user input manually
    st.sidebar.header('Input Parameters')

    daily_time_spent = st.sidebar.number_input("Daily Time Spent on Site (minutes)", min_value=0.0, step=0.1)
    age = st.sidebar.number_input("Age", min_value=0, step=1)
    area_income = st.sidebar.number_input("Area Income ($)", min_value=0.0, step=0.1)
    daily_internet_usage = st.sidebar.number_input("Daily Internet Usage (minutes)", min_value=0.0, step=0.1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    gender_mapping = {"Male": 1, "Female": 0}
    gender = gender_mapping[gender]

    # Create a numpy array with the user inputs
    features = np.array([[daily_time_spent, age, area_income, daily_internet_usage, gender]])
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.write("User is likely to click on the ad.")
    else:
        st.write("User is not likely to click on the ad.")

if __name__ == '__main__':
    main()
