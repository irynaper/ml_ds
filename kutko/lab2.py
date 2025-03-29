import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycaret.datasets import get_data

data = get_data('traffic')

# Опис ознак
st.write("""
### Опис ознак датасету "traffic"
- **holiday**: вказує на святковий день (наприклад, "None" — немає свята, або назва свята).
- **temp**: температура повітря в градусах.
- **rain_1h**: кількість опадів (дощу) за останню годину в мм.
- **snow_1h**: кількість снігу за останню годину в мм.
- **clouds_all**: відсоток хмарності (0–100%).
- **weather_main**: основний тип погоди (наприклад, "Clouds", "Clear", "Rain" і тд.).
- **Rush Hour**: година пік (1 — так, 0 — ні).
- **traffic_volume**: обсяг трафіку (цільова змінна).
""")

# Візуалізація 1: Гістограма
st.subheader('Гістограма розподілу обсягу трафіку')
fig, ax = plt.subplots()
sns.histplot(data['traffic_volume'], bins=30, kde=True, ax=ax)
ax.set_xlabel('Обсяг трафіку')
ax.set_ylabel('Частота')
st.pyplot(fig)

# Візуалізація 2: Стовпчикова діаграма
st.subheader('Середній обсяг трафіку за типом погоди')
weather_group = data.groupby('weather_main')['traffic_volume'].mean().reset_index()
fig = px.bar(weather_group, x='weather_main', y='traffic_volume', title='Середній обсяг трафіку за типом погоди')
st.plotly_chart(fig)

# Візуалізація 3: Боксплот
st.subheader('Боксплот обсягу трафіку за годиною пік')
fig, ax = plt.subplots()
sns.boxplot(x='Rush Hour', y='traffic_volume', data=data, ax=ax)
ax.set_xlabel('Година пік (1 - так, 0 - ні)')
ax.set_ylabel('Обсяг трафіку')
st.pyplot(fig)

# Візуалізація 4: Інтерактивний Scatter Plot
st.subheader('Scatter Plot: Обсяг трафіку vs вибрана ознака')
feature = st.selectbox('Виберіть ознаку для осі X', ['temp', 'rain_1h', 'snow_1h', 'clouds_all'])
fig = px.scatter(data, x=feature, y='traffic_volume', title=f'Обсяг трафіку vs {feature}')
st.plotly_chart(fig)

# Завантаження моделі та прогнозування
model = load_model('traffic_model')
predictions = predict_model(model, data=data)

prediction_column = predictions.columns[-1]
# Обчислення метрик
st.subheader("Model Metrics")
if prediction_column in predictions.columns:
    mae = mean_absolute_error(data['traffic_volume'], predictions[prediction_column])
    mse = mean_squared_error(data['traffic_volume'], predictions[prediction_column])
    r2 = r2_score(data['traffic_volume'], predictions[prediction_column])
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
else:
    st.error("Prediction column not found in the predictions DataFrame.")

# Графік прогнозів vs реальних значень
st.subheader('Графік прогнозів vs реальних значень')
fig, ax = plt.subplots()
ax.scatter(data['traffic_volume'], predictions[prediction_column], alpha=0.5)
ax.plot([data['traffic_volume'].min(), data['traffic_volume'].max()], 
        [data['traffic_volume'].min(), data['traffic_volume'].max()], 'r--')
ax.set_xlabel('Реальний обсяг трафіку')
ax.set_ylabel('Прогнозований обсяг трафіку')
st.pyplot(fig)