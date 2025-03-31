import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.classification import load_model, predict_model

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.markdown("""# Аналіз раку молочної залози""")

# 1. Завантаження даних
st.title("Датасет Cancer")
data = get_data('cancer')
st.write("Перші 5 рядків датасету:")
st.write(data.head())

# 2. Опис ознак
st.subheader("Опис ознак датасету 'cancer'")
description = {
    "Class": "Клас пухлини (0 – доброякісна, 1 – злоякісна)",
    "age": "Вік пацієнта",
    "menopause": "Стадія менопаузи",
    "tumor-size": "Розмір пухлини",
    "inv-nodes": "Кількість уражених лімфовузлів",
    "node-caps": "Наявність капсули вузла",
    "deg-malig": "Ступінь злоякісності",
    "breast": "Уражена сторона грудей",
    "breast-quad": "Квадрант ураження грудей",
    "irradiat": "Попереднє опромінення"
}
st.table(pd.DataFrame.from_dict(description, orient='index', columns=['Опис']))

# 3. Візуалізація даних
st.header("Візуалізація ознак")

# Розподіл класів пухлин
st.subheader("Розподіл доброякісних та злоякісних пухлин")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=data, palette='viridis', ax=ax)
ax.set_xlabel("Клас пухлини")
ax.set_ylabel("Кількість пацієнтів")
st.pyplot(fig)

# Boxplot розміру пухлини
st.subheader("Розмір пухлини")
fig, ax = plt.subplots()
sns.boxplot(y=data['tumor-size'], ax=ax)
st.pyplot(fig)

# Кореляційна матриця
st.subheader("Кореляційна матриця")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

# 4. Завантаження моделі та передбачення
st.header("Результати моделі")
model = load_model('cancer_best_model')
predictions = predict_model(model, data=data)

# Вивід метрик
st.write("Перші передбачення:")
st.write(predictions.head())

# Матриця плутанини
st.subheader("Матриця неточностей")
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(predictions['Class'], predictions['prediction_label']), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Передбачене")
ax.set_ylabel("Реальне")
st.pyplot(fig)

# Розподіл ймовірностей
st.subheader("Розподіл ймовірностей передбачення")
fig, ax = plt.subplots()
sns.histplot(predictions['prediction_score'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Ймовірність злоякісної пухлини")
ax.set_ylabel("Кількість")
st.pyplot(fig)

# Boxplot залишків
st.subheader("Boxplot залишків (помилок)")
fig, ax = plt.subplots()
sns.boxplot(y=predictions['prediction_score'] - predictions['Class'], ax=ax)
ax.set_ylabel("Залишки")
st.pyplot(fig)

# Висновок
st.header("Висновок")
st.write("У результаті аналізу даних пацієнтів з раком молочної залози було побудовано візуалізації, а також використано навчено модель для передбачення злоякісності пухлин. Результати моделі представлені у вигляді метрик і графіків.")
