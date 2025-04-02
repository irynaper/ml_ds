
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data

# Заголовок
st.title("Прогноз: Чи буде гравець в NBA через 5 років?")
st.write("Введіть статистику гравця, щоб передбачити результат")

# Завантаження моделі
model = load_model('nba_classifier_model')

# Отримання даних та визначення ознак
nba = get_data("nba")
nba.dropna(inplace=True)

# Всі ознаки крім цільової
features = nba.drop(columns=["TARGET_5Yrs"])
input_data = {}

# Динамічне створення полів вводу
for col in features.columns:
    if pd.api.types.is_numeric_dtype(nba[col]):
        val = float(nba[col].mean())
        input_data[col] = st.number_input(col, value=val)
    else:
        options = list(nba[col].unique())
        input_data[col] = st.selectbox(col, options)

# Перетворення у DataFrame
input_df = pd.DataFrame([input_data])

# Кнопка для прогнозу
if st.button("Прогнозувати"):
    prediction = predict_model(model, data=input_df)
    result = prediction.iloc[0]["prediction_label"] if "prediction_label" in prediction.columns else prediction.iloc[0]["Label"]
    st.success(f"Прогноз: {result}")
