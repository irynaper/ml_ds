
import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження моделі
model = load_model('nba_classifier_model')

st.title("Прогноз позиції гравця NBA")
st.write("Введіть статистику гравця для передбачення його позиції")

PTS = st.number_input("Очки за гру (PTS)", 0, 100, 20)
AST = st.number_input("Передачі за гру (AST)", 0, 20, 5)
TRB = st.number_input("Підбирання за гру (TRB)", 0, 20, 7)
STL = st.number_input("Перехоплення (STL)", 0, 10, 2)

input_df = pd.DataFrame({
    'PTS': [PTS],
    'AST': [AST],
    'TRB': [TRB],
    'STL': [STL],
})

if st.button("Прогнозувати позицію"):
    prediction = predict_model(model, data=input_df)
    st.success(f"Ймовірна позиція гравця: {prediction['Label'][0]}")

# Опис ознак
st.subheader("Опис ознак")
st.write("""
- **PTS** — кількість очок, які гравець набирає в середньому за гру  
- **AST** — кількість результативних передач за гру  
- **TRB** — підбирання за гру  
- **STL** — перехоплення за гру
""")

# Візуалізація
st.subheader("Гістограма очок (PTS)")
nba = pd.read_csv("nba.csv")
nba.dropna(inplace=True)
fig, ax = plt.subplots()
ax.hist(nba['PTS'], bins=20)
ax.set_title("Розподіл очок серед гравців")
ax.set_xlabel("Очки")
ax.set_ylabel("Кількість гравців")
st.pyplot(fig)

# Метрики моделі
st.subheader("Метрики моделі")
from pycaret.classification import pull
metrics = pull()
st.dataframe(metrics)
