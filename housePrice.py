import streamlit as st
import pandas as pd 
import numpy as np 
import joblib
import warnings 
warnings.filterwarnings('ignore')

# import data 
data = pd.read_csv('USA_Housing.csv')

# import model 
model = joblib.load('housePredictor(1).pkl')

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>HOUSE PRICE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By GomyCode MTW</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com-8.png')

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)

st.markdown("<p style = 'text-align: justify'>The predictive house price modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting the market value of residential properties. By analyzing historical data, identifying key features influencing house prices, and employing advanced regression algorithms, the project seeks to provide valuable insights for homebuyers, sellers, and real estate professionals. The primary objective of this project is to create a reliable machine learning model that accurately predicts house prices based on relevant features such as location, size, number of bedrooms, amenities, and other influencing factors. The model should be versatile enough to adapt to different real estate markets, providing meaningful predictions for a wide range of properties.", unsafe_allow_html= True)

st.sidebar.image('pngwing.com-6.png', caption= 'Welcome User')

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)


input_choice = st.sidebar.radio('Choose Your Input Type', ['Slider Input', 'Number Input'])

if input_choice == 'Slider Input':
    area_income = st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.slider('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.slider('Average Number Of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.slider('Number Of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max())

else:
    area_income = st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.number_input('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.number_input('Average Number Of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.number_input('Number Of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max())

input_vars = pd.DataFrame({'Avg. Area Income': [area_income], 
                           'Avg. Area House Age': [house_age],
                           'Avg. Area Number of Rooms': [room_num],
                           'Avg. Area Number of Bedrooms': [bedrooms],
                            'Area Population': [population]
                           })

st.markdown("<hr ", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style = 'margin: -30px; color: olive; font-family: helvetica '>User Input Variables</h5>", unsafe_allow_html = True)
st.dataframe(input_vars)

predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')
