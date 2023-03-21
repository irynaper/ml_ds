import streamlit as st
import pandas as pd
from pycaret.datasets import get_data
import plotly.express as px
import joblib

# Загрузка дата сета
glass = get_data('glass')

# Загрузка модели
model = joblib.load('glass_model.pkl')

# Название графика (тайтл)
st.set_page_config(page_title='Glass Dataset Visualization', page_icon=':bar_chart:', layout='wide')

# Панель для выбора графика
chart_type = st.sidebar.selectbox('Select Chart Type:', ['Histogram', 'Boxplot', 'Bar Chart', 'Scatter Plot'])
feature = st.sidebar.selectbox('Select Feature:', glass.columns[:-1])

# Функция для выбора графика
def generate_chart(chart_type):
    if chart_type == 'Histogram':
        # Создание гистограммы
        st.subheader(f'Histogram of {feature}')
        fig = px.histogram(glass, x=feature)
        st.plotly_chart(fig.to_dict())

    elif chart_type == 'Boxplot':
        # Создание Box plot для выбраного элемента, сгрупированного за типом стекла
        st.subheader(f'Boxplot of {feature} by Glass Type')
        fig = px.box(glass, x='Type', y=feature)
        st.plotly_chart(fig.to_dict())

    elif chart_type == 'Bar Chart':
        # Создание гистограмы с накоплением для выбранного объекта за типом стекла
        st.subheader(f'Stacked Bar Chart of {feature} by Glass Type')
        chart_data = glass.groupby('Type')[feature].mean().reset_index()
        chart_data = chart_data.melt(id_vars='Type', var_name='Feature', value_name='Value')
        fig = px.bar(chart_data, x='Type', y='Value', color='Feature', barmode='stack')
        st.plotly_chart(fig.to_dict())

    elif chart_type == 'Scatter Plot':
        # Создание Scatter plot для двух признаков, с цветом согласно результатам предсказания модели
        st.subheader(f'Scatter Plot of {feature} by Index with Prediction')
        chart_data = glass[[feature, 'Type']]
        chart_data['Prediction'] = model.predict(glass.drop('Type', axis=1))
        fig = px.scatter(chart_data, x=feature, y=chart_data.index, color='Prediction', template='simple_white')
        st.plotly_chart(fig.to_dict())

# Отображение графика
generate_chart(chart_type)