# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:01:19 2025

@author: smirn
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.classification import load_model, predict_model

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# 1. Завантаження даних
st.title(" Аналіз президентських виборів у США")
data = get_data('us_presidential_election_results')
st.write("Перші 5 рядків датасету:")
st.write(data.head())

# 2. Опис ознак
st.subheader("Опис ознак датасету")
description = {
    'cycle': 'Рік виборчого циклу',
    'state': 'Штат',
    'dem_poll_avg': 'Середній рейтинг демократів у опитуваннях (%)',
    'dem_poll_avg_margin': 'Відстань демократів від республіканців у опитуваннях (%-п.)',
    'incumbent_party': 'Партія чинного президента',
    'incumbent_running': 'Чинний президент балотується (1 - так, 0 - ні)',
    'party_winner': 'Переможна партія у штаті'
}
st.table(pd.DataFrame.from_dict(description, orient='index', columns=['Опис']))

# 3. Візуалізація даних
st.header("📊 Візуалізація даних")

# Фільтрація за роком
selected_cycle = st.select_slider("Оберіть рік виборів:", options=sorted(data['cycle'].unique()))
filtered_data = data[data['cycle'] == selected_cycle]

# Розподіл переможців по штатах
st.subheader(f"Розподіл переможців по штатах у {selected_cycle} році")
fig, ax = plt.subplots(figsize=(10, 6))
party_dist = filtered_data['party_winner'].value_counts()
sns.barplot(x=party_dist.index, y=party_dist.values, ax=ax)
ax.set_xlabel("Партія")
ax.set_ylabel("Кількість штатів")
ax.set_title(f"Розподіл перемог по партіях у {selected_cycle} році")
st.pyplot(fig)

# Теплокарта середніх рейтингів демократів
st.subheader(f"Топ штатів за рейтингом демократів у {selected_cycle} році")
top_states = filtered_data.nlargest(10, 'dem_poll_avg')[['state', 'dem_poll_avg']]
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='dem_poll_avg', y='state', data=top_states, ax=ax)
ax.set_xlabel("Середній рейтинг демократів (%)")
ax.set_ylabel("Штат")
st.pyplot(fig)

# Вплив чинного президента
st.subheader("Вплив чинного президента на результати")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Графік 1: Чинний президент балотується
incumbent_running = filtered_data.groupby(['incumbent_running', 'party_winner']).size().unstack()
incumbent_running.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_xlabel("Чинний президент балотується")
ax1.set_ylabel("Кількість штатів")
ax1.set_xticks([0, 1], ['Ні', 'Так'], rotation=0)

# Графік 2: Партія чинного президента
incumbent_party = filtered_data.groupby(['incumbent_party', 'party_winner']).size().unstack()
incumbent_party.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_xlabel("Партія чинного президента")
ax2.set_ylabel("Кількість штатів")

plt.tight_layout()
st.pyplot(fig)

# Інтерактивна фільтрація за штатом
selected_state = st.selectbox("Оберіть штат для аналізу:", options=data['state'].unique())
state_data = data[data['state'] == selected_state]

# Історична динаміка для обраного штату
st.subheader(f"Історична динаміка для штату {selected_state}")
fig, ax1 = plt.subplots(figsize=(12, 6))

# Графік рейтингу демократів
ax1.plot(state_data['cycle'], state_data['dem_poll_avg'], 'b-', label='Рейтинг демократів')
ax1.set_xlabel("Рік")
ax1.set_ylabel("Рейтинг демократів (%)", color='b')
ax1.tick_params('y', colors='b')

# Графік відстані
ax2 = ax1.twinx()
ax2.plot(state_data['cycle'], state_data['dem_poll_avg_margin'], 'r--', label='Відстань від республіканців')
ax2.set_ylabel("Відстань (%-п.)", color='r')
ax2.tick_params('y', colors='r')

# Додавання переможців
for idx, row in state_data.iterrows():
    color = 'blue' if row['party_winner'] == 'democrat' else 'red'
    ax1.axvline(x=row['cycle'], color=color, alpha=0.2)

fig.tight_layout()
st.pyplot(fig)

# 4. Завантаження моделі та передбачення
try:
    st.header(" Прогнозування результатів")
    model = load_model('election_poll_predictor')
    
    # Вибір даних для прогнозу
    st.subheader("Введіть параметри для прогнозування:")
    col1, col2, col3 = st.columns(3)
    with col1:
        dem_poll = st.slider("Рейтинг демократів (%)", 0.0, 100.0, 50.0)
    with col2:
        dem_margin = st.slider("Відстань від республіканців (%-п.)", -50.0, 50.0, 0.0)
    with col3:
        incumbent_run = st.selectbox("Чинний президент балотується?", [0, 1])
    
    # Створення даних для прогнозу
    input_data = pd.DataFrame({
        'dem_poll_avg': [dem_poll],
        'dem_poll_avg_margin': [dem_margin],
        'incumbent_running': [incumbent_run]
    })
    
    # Прогнозування
    prediction = predict_model(model, data=input_data)
    winner = "Демократи" if prediction['prediction_label'][0] == 'democrat' else "Республіканці"
    confidence = prediction['prediction_score'][0] * 100
    
    st.success(f"Прогнозований переможець: {winner} (впевненість: {confidence:.1f}%)")
    
except Exception as e:
    st.warning(f"Модель не знайдена або виникла помилка: {str(e)}")

# Висновок
st.header("Висновок")
st.write("""
У ході аналізу даних про президентські вибори в США було досліджено:
- Розподіл перемог між партіями по роках і штатах
- Динаміку рейтингів кандидатів
- Вплив чинного президента на результати виборів
- Історичні тенденції для окремих штатів

Додаток дозволяє інтерактивно досліджувати залежності між різними факторами виборчого процесу.
""")