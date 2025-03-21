import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.classification import load_model, predict_model

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.markdown("""# ПЗ №2. Манжола Богдан. Варіант №23.""")

# 1. Завантаження даних
st.title("📊 Датасет Credit")
data = get_data('credit')
st.write("Перші 5 рядків датасету:")
st.write(data.head())

# 2. Опис ознак
st.subheader("Опис ознак датасету 'credit'")
description = {
    'LIMIT_BAL': 'Ліміт кредиту',
    'SEX': 'Стать (1 = чоловік, 2 = жінка)',
    'EDUCATION': 'Рівень освіти',
    'MARRIAGE': 'Сімейний стан',
    'AGE': 'Вік',
    'PAY_1 - PAY_6': 'Історія платежів',
    'BILL_AMT1-6': 'Сума рахунку за попередні місяці',
    'PAY_AMT1-6': 'Сума платежу за попередні місяці',
    'default': 'Цільова змінна: дефолт (1 = так, 0 = ні)'
}
st.table(pd.DataFrame.from_dict(description, orient='index', columns=['Опис']))

# 3. Візуалізація даних
st.header("📊 Візуалізація ознак")

# Гістограма віку
st.subheader("Розподіл віку клієнтів")
fig, ax = plt.subplots()
sns.histplot(data['AGE'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Вік")
ax.set_ylabel("Кількість клієнтів")
st.pyplot(fig)

# Bar plot середнього PAY_AMT1 по статі
st.subheader("Середня сума платежу (PAY_AMT1) по статі")
grouped = data.groupby('SEX')['PAY_AMT1'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x='SEX', y='PAY_AMT1', data=grouped, ax=ax)
ax.set_xlabel("Стать")
ax.set_ylabel("Середня сума платежу")
st.pyplot(fig)

# Pie chart по EDUCATION
st.subheader("Розподіл рівня освіти")
edu_counts = data['EDUCATION'].value_counts()
fig, ax = plt.subplots()
ax.pie(edu_counts, labels=edu_counts.index, autopct='%1.1f%%')
st.pyplot(fig)

# Boxplot BILL_AMT1
st.subheader("Boxplot суми рахунку (BILL_AMT1)")
fig, ax = plt.subplots()
sns.boxplot(y=data['BILL_AMT1'], ax=ax)
st.pyplot(fig)

# Scatter plot
st.subheader("Залежність між лімітом кредиту та віком")
fig, ax = plt.subplots()
sns.scatterplot(x='AGE', y='LIMIT_BAL', data=data, ax=ax)
ax.set_xlabel("Вік")
ax.set_ylabel("Ліміт кредиту")
st.pyplot(fig)

# Інтерактивна фільтрація
st.subheader("Фільтрація за сімейним станом")
selected = st.multiselect("Оберіть статус:", options=data['MARRIAGE'].unique())
filtered_data = data[data['MARRIAGE'].isin(selected)]
st.write(filtered_data)

# 4. Завантаження моделі та передбачення
st.header("📊 Результати моделі")
model = load_model('credit_best_model')
predictions = predict_model(model, data=data)

# Вивід метрик
st.write("Перші передбачення:")
st.write(predictions.head())

# Матриця плутанини
st.subheader("Матриця неточностей")
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(predictions['default'], predictions['prediction_label']), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Передбачене")
ax.set_ylabel("Реальне")
st.pyplot(fig)

# Розподіл ймовірностей
st.subheader("Розподіл ймовірностей передбачення")
fig, ax = plt.subplots()
sns.histplot(predictions['prediction_score'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Ймовірність дефолту")
ax.set_ylabel("Кількість")
st.pyplot(fig)

# Boxplot залишків
st.subheader("Boxplot залишків (помилок)")
fig, ax = plt.subplots()
sns.boxplot(y=predictions['prediction_score'] - predictions['default'], ax=ax)
ax.set_ylabel("Залишки")
st.pyplot(fig)

# Висновок
st.header("Висновок")
st.write("У результаті аналізу даних клієнтів кредитної організації були побудовані різні візуалізації, а також використано навчено модель для передбачення дефолту клієнтів. Результати моделі візуалізовані у вигляді метрик і графіків.")
