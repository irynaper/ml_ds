#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 23:21:46 2025

@author: AlwastDev
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import load_model, predict_model

st.title("Аналіз Boston Housing Dataset")

st.write("## Опис датасету")
st.write("""
The Boston Housing Dataset містить інформацію про ціни на житло у Бостоні.  
Основні характеристики:
- **CRIM** – рівень злочинності  
- **ZN** – частка житлової землі  
- **INDUS** – частка нежитлової землі  
- **RM** – середня кількість кімнат  
- **AGE** – частка старих будинків  
- **TAX** – податкове навантаження  
- **LSTAT** – частка населення з низьким соціальним статусом  
- **MEDV** – медіанна вартість житла (цільова змінна)  
""")

dataset = get_data("boston")

st.write("### Перегляд датасету")
selected_columns = st.multiselect("Оберіть колонки для відображення", dataset.columns.tolist(), default=dataset.columns.tolist())
st.dataframe(dataset[selected_columns].head())

st.write("### Гістограма цін на житло")
fig, ax = plt.subplots()
sns.histplot(dataset["medv"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.write("### Кореляційна матриця")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.write("### Передбачення моделі")
model = load_model("boston_model")

test_data = dataset.drop(columns=["medv"]).sample(5)
predictions = predict_model(model, data=test_data)

st.write(predictions)

