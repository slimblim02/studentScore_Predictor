import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import sklearn
import warnings
warnings.filterwarnings('ignore')
# from xgboost import XGBRegressor
import streamlit as st
import joblib
import pickle

df = pd.read_csv('student_data.csv')
model = joblib.load('StudentGrade_model.pkl')


#-----------------STREAMLIT IMPLEMENTATION----------
st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>STUDENT GRADE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By MercyB Data Scientist</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Problem Statement</h4>", unsafe_allow_html = True)
st.write("The data used is from a Porteguese Secondary School. The data includes academic and personal characteristics of the student as well as final grades. The task is to predict the final grade from the student information(Regressor)")

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(df, use_container_width = True)

st.sidebar.image('pngwing.com (2).png', caption = 'Welcome Dear User')
age = st.sidebar.number_input('age', )

st.sidebar.write(age) #...     To capture what the person input
health = st.sidebar.number_input('Health', )
age = st.sidebar.number_input('Age')
Mjob = st.sidebar.selectbox('Mothers Job', df['Mjob'].unique())
absences = st.sidebar.number_input('Absences')
Fedu = st.sidebar.number_input("Father's Education")
studytime = st.sidebar.number_input('Studytime')
Medu = st.sidebar.number_input("Mother's Education")
reason = st.sidebar.number_input('Reason')
freetime = st.sidebar.number_input('Freetime')
Walc = st.sidebar.number_input('Walc')

st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Input Variable</h4>", unsafe_allow_html = True)

sel_cols = ['age', 'health', 'Mjob', 'absences', 'Fedu', 'studytime', 'Medu', 'reason', 'freetime','Walc','Final_Grade']

inputs_var = pd.DataFrame()
inputs_var['age'] = [age]
inputs_var['health'] = [health]
inputs_var['Mjob'] = [Mjob]
inputs_var['absences'] = [absences]
inputs_var['Fedu'] = [Fedu]
inputs_var['studytime'] = [studytime]
inputs_var['Medu'] = [Medu]
inputs_var['reason'] = [reason]
inputs_var['freetime'] = [freetime]
inputs_var['Walc'] = [Walc]

st.dataframe(inputs_var, use_container_width = True)

mjob_scaler = joblib.load('Mjob_encoder.pkl')

inputs_var['Mjob'] = mjob_scaler.transform(inputs_var['Mjob'])

# Model Prediction
prediction_button = st.button('Predict Grade')
if prediction_button:
    predicted = model.predict(inputs_var)
    st.success(f'The predicted final grade is {predicted[0].astype(int)}')

