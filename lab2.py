# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:42:43 2025

@author: Марина
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.datasets import get_data
from pycaret.classification import load_model, predict_model

st.write("### Частина income датасету")
dataset = get_data('income')
st.write(dataset.head())

st.write("### Опис ознак income датасету")
description = [
    {"Ознака": "age", "Опис": "Вік"},
    {"Ознака": "workclass", "Опис": "Тип зайнятості"},
    {"Ознака": "education", "Опис": "Рівень освіти"},
    {"Ознака": "education-num", "Опис": "Числова оцінка рівня освіти"},
    {"Ознака": "marital-status", "Опис": "Сімейний стан"},
    {"Ознака": "occupation", "Опис": "Професія"},
    {"Ознака": "relationship", "Опис": "Відносини з кимось"},
    {"Ознака": "race", "Опис": "Раса"},
    {"Ознака": "sex", "Опис": "Стать"},
    {"Ознака": "capital-gain", "Опис": "Дохід від капіталу"},
    {"Ознака": "capital-loss", "Опис": "Втрата капіталу"},
    {"Ознака": "hours-per-week", "Опис": "Кількість годин роботи на тиждень"},
    {"Ознака": "native-country", "Опис": "Країна народження"},
    {"Ознака": "income >50K", "Опис": "Доходи більше 50К (0 - ні, 1 - так)"}
]
df_description = pd.DataFrame(description)
st.table(df_description)


st.write("### Гістограма розподіл віку")
fig_age = px.histogram(dataset, x="age", nbins=30)
st.plotly_chart(fig_age)

st.write("### Стовпчикова діаграма розподілу за рівнем освіти")
st.plotly_chart(px.bar(dataset, x="education"))

st.write("### Кругова діаграма за типом зайнятості")
st.plotly_chart(px.pie(dataset, names="workclass"))

st.write("### Боксплот віку залежно від доходу")
st.plotly_chart(px.box(dataset, x="income >50K", y="age"))

st.write("### Гістограма кількості робочих годин на тиждень")
st.plotly_chart(px.histogram(dataset, x="hours-per-week"))

st.write("### Візуалізація розподілу доходів")
fig_income = px.bar(dataset["income >50K"].value_counts(), labels={"index": "Категорія доходу", "value": "Кількість"})
st.plotly_chart(fig_income)



st.write("### Передбачення на основі навченої моделі")
model = load_model('final_lightgbm_model')
predictions = predict_model(model, data = dataset)
st.write(predictions.head())

# Матриця плутанини (Confusion Matrix)
st.subheader("Confusion Matrix (матриця помилок)")
conf_matrix = pd.crosstab(predictions["income >50K"], predictions["prediction_label"])
fig_conf = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True"))
st.plotly_chart(fig_conf, key="conf_matrix")

# Розподіл ймовірностей передбачення
st.subheader("Розподіл ймовірностей передбачення")
fig_prob = px.histogram(predictions, x="prediction_score", nbins=20, marginal="box")
st.plotly_chart(fig_prob, key="prob_dist")

