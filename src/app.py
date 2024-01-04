import streamlit as st
import pandas as pd
import numpy as np
import joblib

# create a title for our app
st.title("Will the patient have a heart attack? :heart::broken_heart:")
# list of emojis can be found here
# https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/emojis/emojis.py


# Pass in the information we will input about the patient
# these are the features we trained our model with
age = st.slider("Input age", 18, 100)
gender = st.select_slider("Input gender", ["Male", "Female"])
cp = st.selectbox("Input chest pain type", ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
trestbps = st.slider("Input Resting Blood Pressure", 50, 200)
chol = st.slide("Input Cholesterol (mg/dl)", 0, 300)
fbs = st.slider("Input fasting blood sugar (mg/dl)", 50, 200)
restecg = st.select_slider("Input resting electrocardiographic results", ["Normal", "Having ST-T Wave Abnormality", "Definite or Probable Estes Criteria"])
thalac = st.number_input("Input Max Heart Rate Achieved")
exang = st.select_slider("Input execrice induced angina", ["Yes", "No"])
#oldpeak = st.number_input("Input ST depression")
#slope = st.select_slider("Input slope", [0, 1, 2, 3])


# this definition takes the data the user inputs and then makes a prediction based on our trained model
def prediction():
    # load the trained model
    model = joblib.load('./model/best_model.joblib')
    
    # code to take input from streamlit and convert to a dataframe so we can make predictions
    row = np.array([age, gender, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak, slope])
    columns = ['age',
               'sex',
               'cp',
               'trestbps',
               'chol',
               'fbs',
               'restecg',
               'thalach',
               'exang',
               'oldpeak',
               'slope']
    X = pd.DataFrame([row], columns=columns)
    X["sex"] = X["sex"].apply(lambda x: 1 if x == "male" else 0)
    X["age"] = pd.to_numeric(X['age'])
    X["cp"] = X["cp"].map({'Typical Angina': 1,'Atypical Angina': 2, 'Non-Anginal Pain': 3,'Asymptomatic': 4})
    X["trestbps"] = pd.to_numeric(X['trestbps'])
    X["chol"] = pd.to_numeric(X['chol'])
    X["fbs"] = pd.to_numeric(X['fbs'])    
    X["fbs"] = X["fbs"].apply(lambda x: 1 if x > 120 else 0)
    X["restecg"] = X["restecg"].map({'Normal': 0,'Having ST-T Wave Abnormality': 1, 'Definite or Probable Estes Criteria': 2})
    X["thalach"] = pd.to_numeric(X['thalach'])
    X["exang"] = X["exang"].apply(lambda x: 1 if x == "Yes" else 0)
    #X["oldpeak"] = pd.to_numeric(X['oldpeak'])
    #X["slope"] = pd.to_numeric(X['slope'])
    X["oldpeak"] = 1
    X["slope"] = 2
    
    prediction = model.predict(X)[0]
    
    # if the patient is predicted as having a heart attack send a comforting message
    if prediction == 1:
        st.error("I am sorry based on this patient's data they will have a heart attack :broken_heart:")
    else:
        st.success("Based on this patient's data the patient is healthy :heart:")

# create a button that will activate the prediction function when the user clicks on it
st.button("Click here for Prediction", on_click = prediction)

# Resource: https://www.youtube.com/watch?v=Ebb4gUI2IpQ