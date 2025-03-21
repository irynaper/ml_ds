import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data

# Завантажуємо датасет Titanic
df = get_data("titanic")

# Завантаження моделі
model = load_model("pz1")

feature_description = {
    "PassengerId": "Унікальний ідентифікатор пасажира",
    "Survived": "Чи вижив пасажир (1 - так, 0 - ні)",
    "Pclass": "Клас квитка (1, 2, 3)",
    "Name": "Ім'я пасажира",
    "Sex": "Стать (male, female)",
    "Age": "Вік пасажира",
    "SibSp": "Кількість братів/сестер або чоловіка/дружини на борту",
    "Parch": "Кількість батьків/дітей на борту",
    "Ticket": "Номер квитка",
    "Fare": "Вартість квитка",
    "Cabin": "Номер каюти (якщо є)",
    "Embarked": "Порт посадки (C = Cherbourg, Q = Queenstown, S = Southampton)"
}

# Streamlit UI
st.title("Аналіз та Візуалізація Titanic Dataset")

# Відображення опису ознак
st.header("Опис ознак")
st.table(pd.DataFrame(feature_description.items(), columns=["Ознака", "Опис"]))

# Відображення даних
display_rows = st.slider("Кількість рядків для перегляду:", min_value=5, max_value=50, value=10)
st.dataframe(df.head(display_rows))

# Візуалізація розподілу виживших
st.header("Розподіл виживших")
fig, ax = plt.subplots()
sns.countplot(x="Survived", data=df, ax=ax, palette="coolwarm")
ax.set_title("Кількість пасажирів за статусом виживання")
st.pyplot(fig)

# Взаємозв'язок класу та виживання
st.header("Виживання в залежності від класу квитка")
fig, ax = plt.subplots()
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax, palette="coolwarm")
ax.set_title("Виживання за класом квитка")
st.pyplot(fig)

# Інтерактивна фільтрація статі для графіку
sex_filter = st.selectbox("Оберіть стать", df["Sex"].unique())
st.header(f"Виживання серед {sex_filter}")
fig, ax = plt.subplots()
sns.barplot(x="Pclass", y="Survived", data=df[df["Sex"] == sex_filter], ax=ax, palette="coolwarm")
ax.set_title(f"Виживання за класом серед {sex_filter}")
st.pyplot(fig)

# Завантаження моделі та прогнозування
st.header("Прогнозування виживання")
# model = load_model("best_titanic_model")  # Завантаження моделі
if st.button("Зробити прогноз"):
    predictions = predict_model(model, data=df)
    st.dataframe(predictions[["PassengerId", "Survived", "prediction_label"]].head(10))

st.write("Робота виконана з використанням Streamlit та PyCaret.")

