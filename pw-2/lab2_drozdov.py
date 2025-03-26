# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:18:10 2025

@author: Danylo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CSS
st.markdown("""
    <style>
    /* Стиль заголовків */
    h1 {
        color: #2E8B57;
        text-align: center;
        font-size: 36px;
    }
    h2, h3 {
        color: #4682B4;
    }
    h6 {
        color: #2E8B57;
        text-align: right;
    }
    /* Фон сторінки */
    body {
        background-color: #F8F8F8;
    }
    /* Стиль для таблиць */
    table {
        width: 100%;
        background-color: white;
        border-radius: 10px;
        border-collapse: collapse;
    }
    th, td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
        text-align: left;
    }
    th {
        background-color: #4682B4;
        color: white;
    }
    /* Кастомізація sidebar */
    .css-1d391kg {
        background-color: #F0F8FF;
    }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv("Cancer.csv")

st.title("Аналіз датасету Cancer")
st.write("###### Виконав студент КНТ-21-6 Дроздов Данило")

st.write("### 📝 Опис ознак датасету Breast Cancer")
feature_desc = {
    "Clump Thickness": "Товщина скупчень клітин (1-10)",
    "Uniformity of Cell Size": "Однорідність розміру клітин (1-10)",
    "Uniformity of Cell Shape": "Однорідність форми клітин (1-10)",
    "Marginal Adhesion": "Прилипання клітин (1-10)",
    "Single Epithelial Cell Size": "Розмір одиночних епітеліальних клітин (1-10)",
    "Bare Nuclei": "Відсутність ядра (1-10)",
    "Bland Chromatin": "Характеристика хроматину (1-10)",
    "Normal Nucleoli": "Кількість нуклеол у ядрі (1-10)",
    "Mitoses": "Кількість мітозів (1-10)",
    "Class": "Діагноз (2 – доброякісна, 4 – злоякісна пухлина)"
}

st.table(pd.DataFrame(list(feature_desc.items()), columns=["Ознака", "Опис"]))

st.write("### Перші 5 рядків датасету")
st.write(df.head())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("### Прогнозування за допомогою навченої моделі")
predictions_df = pd.DataFrame(X_test.copy())
predictions_df['Class'] = y_test
predictions_df['Predicted Label'] = y_pred

st.write(predictions_df.head())

st.write("### Метрики моделі")
metrics = classification_report(y_test, y_pred, output_dict=True)

metrics_df = pd.DataFrame(metrics).T
st.write(metrics_df)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pd.DataFrame(metrics).iloc[:-1, :].T, annot=True, cmap="Blues", cbar=False, ax=ax)
ax.set_title("Метрики моделі")
st.pyplot(fig)

st.write("### Розподіл цільової змінної")
chart_type = st.radio("Тип графіка для цільової змінної", ["Стовпчиковий", "Кругова діаграма"])

if chart_type == "Стовпчиковий":
    fig, ax = plt.subplots()
    df["Class"].value_counts().plot(kind="bar", color=["blue", "orange"], ax=ax)
    ax.set_title("Розподіл цільової змінної (Class)")
    ax.set_xlabel("Клас (2 – доброякісна, 4 – злоякісна)")
    ax.set_ylabel("Кількість")
    st.pyplot(fig)

elif chart_type == "Кругова діаграма":
    fig, ax = plt.subplots()
    df["Class"].value_counts().plot(kind="pie", autopct='%1.1f%%', colors=["blue", "orange"], ax=ax)
    ax.set_title("Розподіл цільової змінної (Class)")
    ax.set_ylabel("")
    st.pyplot(fig)

st.write("### Боксплоти характеристик")
selected_features = st.multiselect("Оберіть характеристики для боксплотів", df.columns[:-1])

if selected_features:
    fig, ax = plt.subplots(figsize=(10, 6))
    df[selected_features].boxplot(ax=ax)
    ax.set_title("Розподіл значень характеристик")
    ax.set_xticklabels(selected_features)
    st.pyplot(fig)

st.write("# Аналіз завершено! Оберіть інші параметри для дослідження.")
