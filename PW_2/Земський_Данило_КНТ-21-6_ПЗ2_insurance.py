import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import load_model, predict_model

# Завантаження даних
data = get_data('insurance')

# Опис ознак
st.title("Аналіз датасету Insurance")

st.write("### Опис ознак:")
feature_description = {
    "age": "Вік клієнта (років)",
    "sex": "Стать клієнта (male / female)",
    "bmi": "Індекс маси тіла (kg/m²)",
    "children": "Кількість дітей у клієнта",
    "smoker": "Чи є клієнт курцем (yes / no)",
    "region": "Регіон проживання (northeast, northwest, southeast, southwest)",
    "charges": "Витрати на медичне страхування (долари)"
}

st.table(pd.DataFrame(list(feature_description.items()), columns=["Ознака", "Опис"]))

# Інтерактивна візуалізація
st.write("### Візуалізація даних")
selected_feature = st.selectbox("Оберіть ознаку для відображення", data.columns)

fig, ax = plt.subplots()

if data[selected_feature].dtype == 'object':
    data[selected_feature].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'Розподіл {selected_feature}')
else:
    data[selected_feature].plot(kind='hist', bins=20, ax=ax, color='lightgreen', edgecolor='black')
    ax.set_title(f'Гістограма {selected_feature}')

st.pyplot(fig)

# Вивід статистики
st.write("### Описова статистика")
st.write(data.describe())

# Завантаження моделі та передбачення
st.header("Результати моделі")
model = load_model('insurance_model')
predictions = predict_model(model, data=data)

# Вивід метрик
st.write("Перші передбачення:")
st.write(predictions.head())

# Розподіл ймовірностей
st.subheader("Розподіл передбачень")
fig, ax = plt.subplots()
sns.histplot(predictions['prediction_label'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Передбачені витрати")
ax.set_ylabel("Кількість")
st.pyplot(fig)

# Boxplot залишків
st.subheader("Boxplot залишків")
fig, ax = plt.subplots()
sns.boxplot(y=predictions['prediction_score'] - predictions['charges'], ax=ax)
ax.set_ylabel("Залишки")
st.pyplot(fig)