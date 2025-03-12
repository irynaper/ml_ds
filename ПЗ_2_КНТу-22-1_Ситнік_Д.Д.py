import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Додано імпорт
import numpy as np

# Загрузка датасета
df = get_data('airquality')

# Преобразование столбца 'Date' в тип datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Удаление столбцов, которые не могут быть использованы для корреляции (например, Date, Time)
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Отображение описания полей датасету
st.write("Опис полів датасету:")
description = {
    "Date": "Дата вимірювання, у форматі YYYY-MM-DD.",
    "Time": "Час вимірювання, у форматі HH:MM:SS.",
    "CO(GT)": "Концентрація вуглекислого газу (CO) у мікрограмах на кубічний метр (µg/m³).",
    "PT08.S1(CO)": "Резистивність сенсора, що вимірює концентрацію CO (прямо пропорційна концентрації CO).",
    "NMHC(GT)": "Концентрація неполярних органічних сполук (NMHC) у мікрограмах на кубічний метр (µg/m³).",
    "C6H6(GT)": "Концентрація бензолу (C6H6) у мікрограмах на кубічний метр (µg/m³).",
    "PT08.S2(NMHC)": "Резистивність сенсора, що вимірює концентрацію NMHC (прямо пропорційна концентрації NMHC).",
    "NOx(GT)": "Концентрація оксидів азоту (NOx) у мікрограмах на кубічний метр (µg/m³).",
    "PT08.S3(NOx)": "Резистивність сенсора, що вимірює концентрацію NOx (прямо пропорційна концентрації NOx).",
    "NO2(GT)": "Концентрація діоксиду азоту (NO2) у мікрограмах на кубічний метр (µg/m³).",
    "PT08.S4(NO2)": "Резистивність сенсора, що вимірює концентрацію NO2 (прямо пропорційна концентрації NO2).",
    "PT08.S5(O3)": "Резистивність сенсора, що вимірює концентрацію озону (O3) (прямо пропорційна концентрації O3).",
    "T": "Температура в градусах Цельсія.",
    "RH": "Вологість повітря (relative humidity) у відсотках (%).",
    "AH": "Абсолютна вологість, вимірювана в г/м³ (grams per cubic meter)."
}
st.write(pd.DataFrame(list(description.items()), columns=["Feature", "Description"]))

# Візуалізація даних
st.write("Гістограми для числових полів датасету:")
fig, ax = plt.subplots(figsize=(10, 6))
df_numeric.hist(ax=ax, bins=20)
st.pyplot(fig)

# Створення графіків за допомогою seaborn
st.write("Графік кореляції між параметрами:")
correlation_matrix = df_numeric.corr()  # теперь работает только для числовых столбцов
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Додаткові графіки
st.write("Боксплоти для числових полів:")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_numeric, ax=ax3)
st.pyplot(fig3)

st.write("Парні графіки для числових полів:")
fig4 = sns.pairplot(df_numeric)
st.pyplot(fig4)

st.write("Графік розсіювання для CO(GT) та PT08.S1(CO):")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='CO(GT)', y='PT08.S1(CO)', data=df, ax=ax5)
st.pyplot(fig5)

st.write("Графік розсіювання для NOx(GT) та PT08.S3(NOx):")
fig6, ax6 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='NOx(GT)', y='PT08.S3(NOx)', data=df, ax=ax6)
st.pyplot(fig6)

st.write("Графік розсіювання для NO2(GT) та PT08.S4(NO2):")
fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='NO2(GT)', y='PT08.S4(NO2)', data=df, ax=ax7)
st.pyplot(fig7)

st.write("Графік розсіювання для C6H6(GT) та PT08.S5(O3):")
fig8, ax8 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='C6H6(GT)', y='PT08.S5(O3)', data=df, ax=ax8)
st.pyplot(fig8)

# Завантаження та використання моделі з першої лабораторної роботи (регресія)
model = load_model('airquality-pipeline')  # Указываем имя модели, если она была сохранена через pycaret

# Прогнозування на тестових даних
predictions = predict_model(model, data=df)

# Відображення результатів моделі
st.write("Результати моделі:")

# Обчислення метрик
mae = mean_absolute_error(df['NOx(GT)'], predictions['prediction_label'])
mse = mean_squared_error(df['NOx(GT)'], predictions['prediction_label'])
rmse = np.sqrt(mse)
r2 = r2_score(df['NOx(GT)'], predictions['prediction_label'])

# Виведення метрик
st.write(f"MAE (Середня абсолютна похибка): {mae:.2f}")
st.write(f"MSE (Середня квадратична похибка): {mse:.2f}")
st.write(f"RMSE (Середньоквадратична похибка): {rmse:.2f}")
st.write(f"R² (Коефіцієнт детермінації): {r2:.2f}")

# Графік фактичних значень vs прогнозованих
st.write("Графік фактичних значень vs прогнозованих:")
fig9, ax9 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['NOx(GT)'], y=predictions['prediction_label'], ax=ax9)
ax9.plot([df['NOx(GT)'].min(), df['NOx(GT)'].max()], 
         [df['NOx(GT)'].min(), df['NOx(GT)'].max()], 
         color='red', linestyle='--')
ax9.set_xlabel("Фактичні значення")
ax9.set_ylabel("Прогнозовані значення")
st.pyplot(fig9)





# Гістограма залишків (різниця між фактичними та прогнозованими значеннями)
st.write("Гістограма залишків:")
residuals = df['NOx(GT)'] - predictions.get('Label', predictions.get('prediction_label'))
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, ax=ax2)
ax2.set_xlabel("Залишки")
ax2.set_ylabel("Частота")
st.pyplot(fig2)

# Графік розподілу фактичних та прогнозованих значень
st.write("Графік розподілу фактичних та прогнозованих значень:")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.kdeplot(df['NOx(GT)'], label="Фактичні значення", ax=ax3)
sns.kdeplot(predictions['prediction_label'], label="Прогнозовані значення", ax=ax3)
ax3.set_xlabel("Значення")
ax3.set_ylabel("Щільність")
ax3.legend()
st.pyplot(fig3)

# Інтерактивний вибір стовпців для візуалізації
st.write("Інтерактивний вибір стовпців для візуалізації:")
selected_column = st.selectbox("Виберіть стовпець для візуалізації:", df_numeric.columns)

# Гістограма для вибраного стовпця
st.write(f"Гістограма для стовпця {selected_column}:")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.histplot(df[selected_column], kde=True, ax=ax4)
st.pyplot(fig4)

# Графік розсіювання з вибором стовпців
st.write("Графік розсіювання з вибором стовпців:")
x_axis = st.selectbox("Виберіть стовпець для осі X:", df_numeric.columns)
y_axis = st.selectbox("Виберіть стовпець для осі Y:", df_numeric.columns)
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax5)
st.pyplot(fig5)