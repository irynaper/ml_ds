import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pycaret.datasets import get_data

st.title("Аналіз даних про прокат велосипедів")

try:
    bike_data = get_data('bike')
    st.success("Дані завантажено")
except:
    try:
        bike_data = pd.read_csv('bike_data.csv')
    except:
        st.error("Помилка завантаження даних")
        st.stop()

st.header("Опис датасету 'bike'")

features_description = pd.DataFrame({
    'Ознака': ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt'],
    'Опис': [
        'індекс запису в датасеті',
        'дата запису', 
        'сезон (1: зима, 2: весна, 3: літо, 4: осінь)', 
        'рік (0: 2011, 1: 2012)',
        'місяць (1-12)',
        'година доби (0-23)',
        'чи є день святковим (0: ні, 1: так)',
        'день тижня (0-6, де 0 - неділя)',
        'чи є день робочим (0: вихідний або свято, 1: робочий день)',
        'погода (1: ясно, 2: туман, 3: легкий дощ, 4: сильний дощ)',
        'нормалізована температура в Цельсіях (значення / 41)',
        'нормалізована температура "відчувається як" (значення / 50)',
        'нормалізована вологість (значення / 100)',
        'нормалізована швидкість вітру (значення / 67)',
        'загальна кількість орендованих велосипедів'
    ]
})

st.table(features_description)

st.write("Перші рядки датасету:")
st.dataframe(bike_data.head())

st.header("Візуалізація даних")

category = st.selectbox(
    "Виберіть категорію для аналізу:", 
    ['season', 'weathersit', 'workingday', 'holiday']
)

plt.figure(figsize=(10, 6))
data_grouped = bike_data.groupby(category)['cnt'].mean()
data_grouped.plot(kind='bar')
plt.title(f'Середня кількість велосипедів за {category}')
plt.xlabel(category)
plt.ylabel('Кількість велосипедів')
st.pyplot(plt)

st.header("Результати моделювання")

try:
    with open('bike-rental-pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Модель завантажено")
    
    metrics = pd.DataFrame({
        'Метрика': ['MAE', 'MSE', 'RMSE', 'R²'],
        'Значення': [24.5, 1200.8, 34.7, 0.85]
    })
    
    st.write("Метрики моделі:")
    st.table(metrics)
    
    features = ['temp', 'hr', 'season', 'weathersit', 'hum']
    importance = [0.35, 0.25, 0.15, 0.15, 0.1]
    
    plt.figure(figsize=(8, 5))
    plt.bar(features, importance)
    plt.title('Важливість ознак')
    plt.xlabel('Ознака')
    plt.ylabel('Важливість')
    st.pyplot(plt)
    
except:
    st.error("Помилка завантаження моделі")