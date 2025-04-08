import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Jewellery Dataset Dashboard", layout="wide")

st.title("Jewellery Dataset Dashboard")
st.markdown("Аналіз даних з PyCaret датасету `jewellery`")

# --- Завантаження датасету ---
@st.cache_data
def load_data():
    return pd.read_csv("jewellery.csv")

df = load_data()

# --- Перегляд даних ---
st.write("### Перші 5 рядків датасету:")
st.dataframe(df.head())

# --- Опис ознак ---
st.write("### Опис ознак:")
st.markdown("""
- **Age** — вік клієнта  
- **Income** — річний дохід клієнта (в доларах)  
- **SpendingScore** — коефіцієнт купівельної активності (від 0 до 1)  
- **Savings** — сума накопичень (в доларах)
""")

# --- Візуалізації ---
st.write("## Візуалізація даних")

col1, col2 = st.columns(2)

with col1:
    num_col = st.selectbox("Оберіть числову колонку для гістограми:", df.select_dtypes("number").columns)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[num_col], kde=True, ax=ax1)
    ax1.set_title(f"Гістограма: {num_col}")
    ax1.set_xlabel(num_col)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, orient="h", ax=ax2)
    ax2.set_title("Boxplot усіх числових ознак")
    st.pyplot(fig2)

# --- Інтерактивна фільтрація ---
st.write("## Інтерактивний фільтр за віком")
age_range = st.slider("Оберіть діапазон віку:", int(df.Age.min()), int(df.Age.max()), (30, 60))
filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
st.dataframe(filtered_df)

# --- Результати моделі (плейсхолдер) ---
st.write("## Результати моделі з першої лабораторної")
# import joblib

# st.write("## Прогноз SpendingScore за допомогою моделі")

# uploaded_model = joblib.load("best_spending_score_model.pkl")
# predictions = uploaded_model.predict(df.drop(columns=["SpendingScore"]))
# df["PredictedScore"] = predictions

# st.write("### Порівняння справжніх і передбачених значень:")
# st.dataframe(df[["SpendingScore", "PredictedScore"]].head())


from pycaret.datasets import get_data
data = get_data('jewellery')
data.to_csv('jewellery.csv', index=False)

from pycaret.regression import *
reg = setup(data=data, target='SpendingScore', session_id=123,
            train_size=0.8, verbose=False)
best_model = compare_models()
evaluate_model(best_model)


model = best_model
predictions = predict_model(model, data=filtered_df)

from sklearn.metrics import mean_squared_error, r2_score
y_true = predictions['SpendingScore']
y_pred = predictions['prediction_label']

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")
fig, ax = plt.subplots()
ax.scatter(y_true, y_pred)
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
ax.set_xlabel("Actual Spending Score")
ax.set_ylabel("Predicted Spending Score")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)


