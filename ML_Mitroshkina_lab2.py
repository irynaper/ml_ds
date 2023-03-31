# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: kristina

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.datasets import get_data
from pycaret.regression import *


forest = get_data('forest')

st.title("Dataset Description")
description = pd.DataFrame({
    'Attribute': forest.columns,
    'Description': [
        'X-axis spatial coordinate within the Montesinho park map',
        'Y-axis spatial coordinate within the Montesinho park map',
        'Month of the year',
        'Day of the week',
        'FFMC index from the FWI system',
        'DMC index from the FWI system',
        'DC index from the FWI system',
        'ISI index from the FWI system',
        'Temperature in Celsius degrees',
        'Relative humidity in %',
        'Wind speed in km/h',
        'Outside rain in mm/m2',
        'The burned area of the forest (in ha)'
    ]
})

st.table(description)

st.title("Visualizing Forest Data with Streamlit")
st.write("This app visualizes the `forest` dataset from pycaret.")
st.write("## Please select the type of visualization below")

# Create a sidebar with options for the user to select
sidebar_options = ['Overview', 'Histogram', 'Bar Chart', 'Pie Chart', 'Box Plot']
selected_option = st.selectbox('## Select a Visualization', sidebar_options)

# Display the selected visualization
if selected_option == 'Overview':
    st.subheader('Dataset Overview')
    st.write(forest.head())
elif selected_option == 'Histogram':
    st.subheader('Histogram of Wind Speed')
    fig, ax = plt.subplots()
    ax.hist(forest['wind'], bins=20)
    ax.set_xlabel('Wind Speed')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
elif selected_option == 'Bar Chart':
    st.subheader('Bar Chart of Month vs. Number of Fires')
    grouped_data = forest.groupby('month')['area'].sum()
    fig, ax = plt.subplots()
    ax.bar(grouped_data.index, grouped_data.values)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Fires')
    st.pyplot(fig)
elif selected_option == 'Pie Chart':
    st.subheader('Pie Chart of Burned Area')
    fig, ax = plt.subplots()
    ax.pie(forest['area'], labels=forest['month'], autopct='%1.1f%%')
    ax.set_title('Burned Area by Month')
    st.pyplot(fig)
else:
    st.subheader('Box Plot of Burned Area')
    fig, ax = plt.subplots()
    ax.boxplot(forest['area'])
    ax.set_ylabel('Burned Area')
    st.pyplot(fig)
    
# Using the model created in lab1 visualize results, metrics etc

st.title("Visualizing trained model from the previos lab")

model = load_model("/Users/admin/Downloads/forest_regression_model.pkl")

st.title('Wildfire Prediction App')
st.write('This app is able to predict fire area based on the trained model from the previos lab and on the input features')

# Add the necessary user inputs and widgets
st.sidebar.title('Input Features')
st.sidebar.write('## For Wildfire Prediction App')
x = pd.DataFrame({
    'X': [st.sidebar.slider('X', 1, 9, 6)],
    'Y': [st.sidebar.slider('Y', 2, 6, 4)],
    'month': [st.sidebar.selectbox('Month', options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])],
    'day': [st.sidebar.selectbox('Day', options=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])],
    'FFMC': [st.sidebar.slider('FFMC', 18.7, 96.2, 90.0)],
    'DMC': [st.sidebar.slider('DMC', 1.1, 291.3, 180.0)],
    'DC': [st.sidebar.slider('DC', 7.9, 860.6, 350.0)],
    'ISI': [st.sidebar.slider('ISI', 0.0, 56.1, 10.0)],
    'temp': [st.sidebar.slider('Temperature', 2.2, 33.3, 20.0)],
    'RH': [st.sidebar.slider('Relative Humidity', 15, 100, 50)],
    'wind': [st.sidebar.slider('Wind Speed', 0.4, 9.4, 4.0)],
    'rain': [st.sidebar.slider('Rainfall', 0.0, 6.4, 0.0)]
})

prediction = model.predict(x)[0]
pred = model.predict(x)

# Visualize the results using Streamlit plots and charts
st.write(f'### Estimated burned area: {prediction:.2f} hectares')

# Create a scatter plot showing the relationship between temperature and the predicted fire area
st.subheader('Relationship between temperature and the predicted fire area')
fig, ax = plt.subplots()
ax.scatter(x['temp'], pred)
ax.set_xlabel('Temperature')
ax.set_ylabel('Predicted fire area')
st.pyplot(fig)


