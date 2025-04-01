import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pycaret.classification import load_model, predict_model

@st.cache_resource
def load_bank_model():
    model = load_model("bank_model.pkl")
    return model
model = load_bank_model()

@st.cache_data
def load_data():
    df = pd.read_csv("bank_data.csv")
    return df

df = load_data()


st.subheader("Визуализация входных данных")
chart_type = st.radio("Выберите тип графика:", ["Гистограмма", "Диограмма стобликом"])


placeholder = st.empty()

if chart_type == "Гистограмма":
    selected_column = st.selectbox("Выберите колонку для гистограммы:", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
    placeholder.pyplot(fig)

elif chart_type == "Диограмма стобликом":
    selected_column = st.selectbox("Выберите колонку для столбчатой диаграммы:", df.columns)

    if df[selected_column].dtype in [np.int64, np.float64]:
        bins = st.slider("Выберите количество интервалов:", min_value=5, max_value=50, value=10)
        df['binned'] = pd.cut(df[selected_column], bins=bins)
        fig, ax = plt.subplots()
        sns.barplot(x=df['binned'].value_counts().index, y=df['binned'].value_counts().values, ax=ax)
        plt.xticks(rotation=45)
        placeholder.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.barplot(x=df[selected_column].value_counts().index, y=df[selected_column].value_counts().values, ax=ax)
        plt.xticks(rotation=45)
        placeholder.pyplot(fig)

st.write("### Метрики модели:")
metrics = pd.read_csv("classification_metrics.csv")
st.dataframe(metrics)
