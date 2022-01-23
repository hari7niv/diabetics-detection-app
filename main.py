from msilib.schema import Error
import streamlit as st
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

st.header("Diabetics dedection app")
data = pd.read_csv('data_diab.csv')
data = data[["Age", "Gender", "Polyuria", "Polydipsia",
             "sudden weight loss", "weakness", "Polyphagia", "Obesity", "class"]]
X = np.array(data.drop(columns='class'))
y = np.array(data['class'])

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(9)
model.fit(X_train, y_train)
Age = st.slider("Age", 15, 100)
Gender = st.selectbox('Gender', ["", "female", 'male'])
Polyuria = st.selectbox(
    'Whether the patient experienced excessive urination or not', ['', "Yes", 'No'])
Polydipsia = st.selectbox(
    'Polydipsia: Whether the patient experienced excessive thirst/excess drinking or not.', ['', "Yes", 'No'])
sudden_weight_loss = st.selectbox(
    'sudden_weight_loss: Whether patient had an episode of sudden weight loss or not.', ['', "Yes", 'No'])
weakness = st.selectbox(
    'weakness: Whether patient had an episode of feeling weak or not', ['', "Yes", 'No'])
Polyphagia = st.selectbox(
    'Polyphagia: Whether patient had an episode of excessive/extreme hunger or not.', ['', "Yes", 'No'])
Obesity = st.selectbox(
    'Obesity: Whether the patient is over weight or not', ['', "Yes", 'No'])

if Gender == "Male" or Gender == "male":
    Gender = 1
if Polyuria == 'yes' or Polyuria == 'Yes':
    Polyuria = 1
if Polydipsia == 'yes' or Polydipsia == "Yes":
    Polydipsia = 1
if sudden_weight_loss == 'yes' or sudden_weight_loss == "Yes":
    sudden_weight_loss = 1
if weakness == 'yes' or weakness == "Yes":
    weakness = 1
if Polyphagia == 'yes' or Polyphagia == "Yes":
    Polyphagia = 1
if Obesity == 'yes' or Obesity == "Yes":
    Obesity = 1

if Gender == "Female" or Gender == "female":
    Gender = 0
if Polyuria == 'no' or Polyuria == 'No':
    Polyuria = 0
if Polydipsia == 'no' or Polydipsia == "No":
    Polydipsia = 0
if sudden_weight_loss == 'no' or sudden_weight_loss == "No":
    sudden_weight_loss = 0
if weakness == 'no' or weakness == "No":
    weakness = 0
if Polyphagia == 'no' or Polyphagia == "No":
    Polyphagia = 0
if Obesity == 'no' or Obesity == "No":
    Obesity = 0
else:
    Age, Gender, Polyuria, Polydipsia,sudden_weight_loss, weakness, Polyphagia, Obesity = 0
    

y_pred = model.predict([[Age, Gender, Polyuria, Polydipsia,
                       sudden_weight_loss, weakness, Polyphagia, Obesity]])
for x in range(len(y_pred)):
        print(y_pred[x])
        if y_pred[x] == 1:
            st.text("Negative")
        if y_pred[x] == 0:
            st.text("Possitive")
