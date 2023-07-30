import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st

insurance_dataset = pd.read_csv('insurance.csv')


# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)




X = insurance_dataset.drop(columns='charges', axis=1)
X=X.drop(columns='region', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

st.title('Insurance Price Priction')
st.subheader('by Chandanbir and Deesha')


age=st.number_input('Enter your age', 'Age')
sex_text=st.text_input('Enter your sex ','(M or F)')
bmi=st.number_input('Enter your bmi', 'bmi')
smoker=st.number_input('Do you smoke', 'y or n')
children=st.number_input('Enter the number of children you have')

if (sex_text == 'm' or sex_text == 'male' or sex_text == 'Male'):
    sex=0
else:
    sex=1
    

input_data = ()

input_data.append(age,sex,bmi,children,smoker)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

st.write('The insurance cost is USD ', prediction[0])