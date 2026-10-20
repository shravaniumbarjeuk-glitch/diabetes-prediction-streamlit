import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('diabetes.csv')

# Data Standardization
scaler = StandardScaler()
X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']
scaler.fit(X)
standardized_data = scaler.transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit App
st.title('Diabetes Prediction App')

# Sidebar for user input
st.sidebar.header('Input Features')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=5)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=199, value=166)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=19)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=175)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=25.8)
diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.587)
age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=51)


input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree_function,
    'Age': age
}


input_df = pd.DataFrame([input_data])

# Standardize the input data
std_data = scaler.transform(input_df)


if st.button('Predict'):
    prediction = classifier.predict(std_data)

    
    if prediction[0] == 0:
        st.success('The person is not diabetic')
    else:
        st.error('The person is diabetic')
