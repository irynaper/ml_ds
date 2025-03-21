import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.datasets import get_data


# Завантаження даних
data = get_data("insurance")

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
