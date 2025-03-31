import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pycaret.classification import load_model, predict_model

@st.cache_resource
def load_fire_model():
    model = load_model("prediction_model")
    return model
model = load_fire_model()

@st.cache_data
def load_data():
    df = pd.read_csv("forest_data.csv")
    return df

df = load_data()

temperature = st.slider("temp - температура", min_value=0, max_value=50, value=25)
humidity = st.slider("HR - вологість", min_value=0, max_value=100, value=50)
wind = st.slider("wind - вітер", min_value=0, max_value=20, value=5)
rain = st.slider("rain - опади", min_value=0, max_value=100, value=0)
FFMC = st.slider("FFMC - вологість лісової підстилки", min_value=0, max_value=100, value=50)
DMC = st.slider("DMC - вологість мохової підстилки", min_value=0, max_value=100, value=50)
DC = st.slider("DC - ступінь сушіння", min_value=0, max_value=100, value=50)
ISI = st.slider("ISI - швидкість розв. пожежі", min_value=0, max_value=100, value=5)
input_data = pd.DataFrame([[0, 0, 1, 1, FFMC, DMC, DC, ISI, rain, 1, temperature, humidity, wind]],
                          columns=['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'rain', 'area', 'temp', 'RH', 'wind'])

prediction = predict_model(model, data=input_data)
st.write(f"Результат передбачення: **{prediction['prediction_label'][0]}**")

st.subheader("Візуалізація вхідних даних")
chart_type = st.radio("Оберіть тип графіка:", ["Гістограма", "Стовпчиковий графік"])

if chart_type == "Гістограма":
    selected_column = st.selectbox("Оберіть колонку для гістограми:", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

elif chart_type == "Стовпчиковий графік":
    selected_column = st.selectbox("Оберіть колонку для стовпчикового графіка:", df.columns)

    if df[selected_column].dtype in [np.int64, np.float64]:
        bins = st.slider("Виберіть кількість інтервалів:", min_value=5, max_value=50, value=10)
        df['binned'] = pd.cut(df[selected_column], bins=bins)
        fig, ax = plt.subplots()
        sns.barplot(x=df['binned'].value_counts().index, y=df['binned'].value_counts().values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.barplot(x=df[selected_column].value_counts().index, y=df[selected_column].value_counts().values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

st.write("### Метрики моделі:")
metrics = pd.read_csv("classification_metrics.csv")
st.dataframe(metrics) 