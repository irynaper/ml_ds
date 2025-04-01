import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model, predict_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pycaret.datasets import get_data

dataset = get_data('kiva')

# Опис ознак
st.write("""
### Опис ознак датасету "kiva"
    1. country: Країна позичальника 'Dominican Republic', 'Ecuador', 'Kenya'
    2. en: Особиста історія позичальника 
    3. gender: Стать позичальника (M = чоловік, F = жінка)
    4. loan_amount: Сума кредиту, яка була схвалена та виплачена (числова величина)
    5. nonpayment: Тип кредитора 'partner', 'lender'
    6. sector: Сектор діяльності позичальника (наприклад, 'Retail', 
            'Clothing', 'Food', 'Services', 'Arts', 'Agriculture',
            'Wholesale', 'Manufacturing', 'Transportation', 'Health',
            'Education', 'Personal Use', 'Construction', 'Housing',
            'Entertainment')
    7. status: Статус кредиту (1 = дефолт, тобто не повернуто, 0 = повернуто)
""")

# Візуалізація: Гістограма суми кредиту
st.subheader("Гістограма суми кредиту")
fig, ax = plt.subplots()
sns.histplot(dataset['loan_amount'], ax=ax)
ax.set_title('Розподіл суми кредиту')
ax.set_xlabel('Сума кредиту')
ax.set_ylabel('Частота')
st.pyplot(fig)

# Візуалізація: Стовпчикова діаграма за статтю
st.subheader("Розподіл за статтю")
gender_counts = dataset['gender'].value_counts()
st.bar_chart(gender_counts)

# Візуалізація: Кругова діаграма для секторів
st.subheader("Кругова діаграма секторів")
sector_counts = dataset['sector'].value_counts()
fig, ax = plt.subplots()
ax.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Інтерактивний вибір країни
st.subheader("Вибір країни для аналізу")
countries = dataset['country'].unique()
selected_country = st.selectbox("Оберіть країну:", countries)
country_data = dataset[dataset['country'] == selected_country]
st.write(f"Дані для країни {selected_country}:", country_data.head())

# Візуалізація: Боксплот суми кредиту за статусом
st.subheader("Боксплот суми кредиту за статусом")
fig, ax = plt.subplots()
sns.boxplot(x='status', y='loan_amount', data=dataset, ax=ax)
ax.set_title('Сума кредиту за статусом')
ax.set_xlabel('Статус (0 = повернуто, 1 = дефолт)')
ax.set_ylabel('Сума кредиту')
st.pyplot(fig)

# Завантаження моделі
model = load_model('kiva_model')

# Прогнозування
predictions = predict_model(model, data=dataset)

# Метрики
st.subheader("Метрики моделі")
y_true = dataset['status']
y_pred = predictions['Label'] if 'Label' in predictions else predictions['prediction_label']
accuracy = accuracy_score(y_true, y_pred)
st.write(f"Точність (Accuracy): {accuracy:.2f}")
st.text("Звіт класифікації:\n" + classification_report(y_true, y_pred))

# Матриця помилок
st.subheader("Матриця помилок")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Передбачено')
ax.set_ylabel('Фактичне')
st.pyplot(fig)