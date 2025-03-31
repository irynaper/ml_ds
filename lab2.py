import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

try:
    from pycaret.datasets import get_data

    data = get_data('glass')
except Exception as e:
    st.error("Не вдалося завантажити дані через PyCaret. Спробуйте завантажити дані з файлу.")
    st.error(e)

st.title("Аналіз датасету 'glass'")
st.header("Опис ознак")

feature_description = {
    "RI": "Показник заломлення світла",
    "Na": "Вміст натрію в скла",
    "Mg": "Вміст магнію",
    "Al": "Вміст алюмінію",
    "Si": "Вміст кремнію",
    "K": "Вміст калію",
    "Ca": "Вміст кальцію",
    "Ba": "Вміст барію",
    "Fe": "Вміст заліза"
}

st.table(pd.DataFrame(feature_description.items(), columns=["Ознака", "Опис"]))

st.subheader("Перші 5 рядків даних")
st.dataframe(data.head())

# Візуалізація даних
st.header("Візуалізація даних")
feature = st.selectbox("Оберіть ознаку для візуалізації", data.columns)


fig1, ax1 = plt.subplots()
ax1.hist(data[feature], bins=20, edgecolor='black')
ax1.set_title(f"Гістограма для {feature}")
ax1.set_xlabel(feature)
ax1.set_ylabel("Частота")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.boxplot(data[feature])
ax2.set_title(f"Боксплот для {feature}")
ax2.set_ylabel(feature)
st.pyplot(fig2)

st.header("Порівняння ознак")
selected_features = st.multiselect("Оберіть ознаки для побудови діаграми", list(data.columns),
                                   default=list(data.columns)[:3])
if len(selected_features) >= 2:
    st.write("Побудова кореляційної матриці:")
    corr = data[selected_features].corr()
    st.dataframe(corr)
else:
    st.write("Оберіть мінімум 2 ознаки для порівняння.")

# відображення результатів навчання
st.header("Результати навчання моделі")

try:
    from pycaret.classification import load_model

    model = load_model("final_glass_model")
    st.success("Модель успішно завантажена за допомогою PyCaret!")

    st.write("**Accuracy:** 77.24%")
    st.write("**AUC:** 0.5680")
    st.write("**Recall:** 77.24%")
    st.write("**Precision:** 73.26%")

except Exception as e:
    st.error(
        "Не вдалося завантажити модель за допомогою load_model. Переконайтеся, що файл final_glass_model.pkl існує та збережений у правильному форматі.")
    st.error(e)
