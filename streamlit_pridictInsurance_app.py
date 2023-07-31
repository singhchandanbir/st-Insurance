import numpy as np
import pandas as pd
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

st.title('Insurance Price Prediction')
st.subheader('by Chandanbir and Deesha')

input_data = []
input_data.append(st.number_input('Enter your age',step=1))
sex_text=st.selectbox('Enter your sex',['Select from drop down','Male','Female'],index=0)
if (sex_text=='Male'):
    sex=1
else:
    sex=0
input_data.append(sex)

input_data.append(st.number_input('Enter your bmi'))
smoker=st.selectbox('Do you smoke',['Select from drop down','Yes','No'],index=0,)

if (smoker=='Yes'):
    smoker_val=1
else:
    smoker_val=0
input_data.append(smoker_val)
input_data.append(st.number_input('Enter the number of children you have',step=1))

if (sex_text == 'm' or sex_text == 'male' or sex_text == 'Male'):
    sex=0
else:
    sex=1
    





# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction=[]
prediction = regressor.predict(input_data_reshaped)
# print(prediction)
if st.button('Predict'):
    st.write('The insurance cost is USD ', prediction[0])