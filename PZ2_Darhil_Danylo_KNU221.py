import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from pycaret.datasets import get_data
Data = get_data('nba')
st.markdown("""# ПЗ № 2. Даргіль Данило Олександрович. КНТу-22-1. Варіант №1.""")
st.markdown("""## Датасет NBA:""")
st.write(Data.describe())

st.markdown("""### **Опис датасету NBA**  

Датасет **NBA** містить статистичні показники гравців NBA та використовується для аналізу їхньої продуктивності, а також для прогнозування майбутньої кар'єри. Основна мета – визначити, чи буде гравець залишатися в NBA через **5 років** після початкового сезону.  

---

#### **Загальна структура датасету**
- **Кількість записів:** 1340 (гравці)  
- **Кількість стовпців:** 21  
- **Типи даних:**  
  - 1 текстовий (`Name` – ім'я гравця)  
  - 2 цілочислові (`GP`, `TARGET_5Yrs`)  
  - 18 числових (float) – статистичні показники гравця  

---

### **Опис стовпців**
| **Стовпець**    | **Опис** |
|-----------------|----------|
| **`Name`**      | Ім'я гравця. |
| **`GP`**        | Кількість зіграних матчів (Games Played). |
| **`MIN`**       | Середня кількість хвилин, проведених на майданчику за гру (Minutes Per Game). |
| **`PTS`**       | Середня кількість очок за гру (Points Per Game). |
| **`FGM`**       | Кількість влучень з гри (Field Goals Made). |
| **`FGA`**       | Кількість спроб кидків з гри (Field Goals Attempted). |
| **`FG%`**       | Відсоток влучань з гри (Field Goal Percentage). |
| **`3P Made`**   | Кількість влучних триочкових кидків (Three-Point Field Goals Made). |
| **`3PA`**       | Кількість спроб триочкових кидків (Three-Point Field Goals Attempted). |
| **`3P%`**       | Відсоток влучань триочкових кидків (Three-Point Field Goal Percentage). |
| **`FTM`**       | Кількість влучних штрафних кидків (Free Throws Made). |
| **`FTA`**       | Кількість спроб штрафних кидків (Free Throws Attempted). |
| **`FT%`**       | Відсоток влучань штрафних кидків (Free Throw Percentage). |
| **`OREB`**      | Кількість підбирань у нападі (Offensive Rebounds). |
| **`DREB`**      | Кількість підбирань у захисті (Defensive Rebounds). |
| **`REB`**       | Загальна кількість підбирань (Total Rebounds). |
| **`AST`**       | Кількість передач, що привели до результативного кидка (Assists). |
| **`STL`**       | Кількість перехоплень м'яча (Steals). |
| **`BLK`**       | Кількість блокшотів (Blocks). |
| **`TOV`**       | Кількість втрат м'яча (Turnovers). |
| **`TARGET_5Yrs`** | *Цільова змінна:* - `1` – Гравець продовжить виступати в NBA через 5 років. - `0` – Гравець не буде грати в NBA через 5 років. |

---

### **Призначення датасету**
Датасет використовується для таких задач:
- **Аналіз продуктивності гравців:** Визначення найкращих та найгірших показників.
- **Прогнозування кар'єрного шляху:** Використання `TARGET_5Yrs` як цільової змінної в моделі машинного навчання для передбачення кар'єри гравців.
- **Оцінка важливих факторів:** Виявлення ключових характеристик, які впливають на довгострокову кар'єру в NBA.
""")

# Ініціалізація Streamlit
st.title("📊 Аналіз статистики NBA")

# Візуалізація 1: Гістограма розподілу очок (PTS) +++++
st.subheader("1. Розподіл очок серед гравців")
fig, ax = plt.subplots()
sns.histplot(Data['PTS'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Очки за гру")
ax.set_ylabel("Кількість гравців")
st.pyplot(fig)
st.write("Цей графік показує, як розподіляються очки серед гравців NBA.")

# Візуалізація 2: Середній відсоток влучань (FG%) за категоріями ++++++
st.subheader("2. Середній відсоток влучань за категоріями")
position = st.selectbox("Оберіть категорію", ['GP', 'TARGET_5Yrs'])
data_grouped = Data.groupby(position)['FG%'].mean().reset_index()
fig, ax = plt.subplots(figsize=(15, 7))  # Розширено по горизонталі
sns.barplot(x=data_grouped[position], y=data_grouped['FG%'], ax=ax)
ax.set_xlabel(f"Значення {position}")
ax.set_ylabel("Середній відсоток влучань FG%")
ax.set_xticks(ax.get_xticks()[::1])  # Скорочення підписів
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
st.write("Ця діаграма демонструє відношення середнього відсотку влучань (FG%) відповідно до кількості зіграних матчів або того чи продовжить гравець виступати в NBA через 5 років.")

# Візуалізація 3: Співвідношення триочкових спроб (3PA) до загальної кількості кидків (FGA) +++++++
st.subheader("3. Частка триочкових спроб серед усіх кидків")
fig, ax = plt.subplots()
labels = ['3-Очкові спроби', 'Інші спроби']
data_values = [Data['3PA'].sum(), Data['FGA'].sum() - Data['3PA'].sum()]
ax.pie(data_values, labels=labels, autopct='%1.1f%%')
st.pyplot(fig)
st.write("Ця кругова діаграма показує, яку частку займають триочкові спроби серед усіх спроб кидків.")

# Візуалізація 4: Розподіл оборонних і атакувальних підбирань ++++++
st.subheader("4. Розподіл підбирань")
fig, ax = plt.subplots()
sns.boxplot(data=Data[['OREB', 'DREB']], ax=ax)
ax.set_ylabel("Кількість підбирань")
st.pyplot(fig)
st.write("Ця діаграма показує розподіл атакувальних (OREB) та захисних (DREB) підбирань у гравців NBA.")

# Візуалізація 5: Кількість втрат (TOV) у залежності від зіграних матчів (GP) +++++++
st.subheader("5. Кількість втрат у залежності від кількості матчів")
fig, ax = plt.subplots()
sns.scatterplot(x=Data['GP'], y=Data['TOV'], ax=ax)
ax.set_xlabel("Кількість матчів (GP)")
ax.set_ylabel("Втрати м'яча (TOV)")
st.pyplot(fig)
st.write("Ця діаграма демонструє взаємозв’язок між кількістю матчів та втратами м'яча у гравців.")

# Візуалізація 6: Середні очки у різних групах гравців ++++++
st.subheader("6. Середні очки у різних категоріях")
group = st.radio("Оберіть категорію", ['GP', 'TARGET_5Yrs'])
data_grouped = Data.groupby(group)['PTS'].mean().reset_index()
fig, ax = plt.subplots(figsize=(15, 7)) 
sns.barplot(x=data_grouped[group], y=data_grouped['PTS'], ax=ax)
ax.set_xlabel(f"Значення {group}")
ax.set_ylabel("Середні очки")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
st.write("Ця діаграма показує середню кількість очок, які набирають гравці в залежності від кількості зіграних матчів, або в залежності від того, чи буде гравець виступати в продовж наступних 5 років.")

# Візуалізація 7: Відношення результативних передач до перехоплень ++++++
st.subheader("7. Відношення передач до перехоплень")
fig, ax = plt.subplots()
sns.scatterplot(x=Data['AST'], y=Data['STL'], ax=ax)
ax.set_xlabel("Передачі (AST)")
ax.set_ylabel("Перехоплення (STL)")
st.pyplot(fig)
st.write("Ця діаграма показує, як взаємопов’язані результативні передачі та перехоплення у гравців.")

# Візуалізація 8: Відношення штрафних спроб (FTA) до влучань (FTM) ++++++
st.subheader("8. Відношення штрафних спроб до влучань")
fig, ax = plt.subplots()
labels = ['Штрафні спроби (FTA)', 'Штрафні влучання (FTM)']
data_values = [Data['FTA'].sum(), Data['FTM'].sum()]
sns.barplot(x=labels, y=data_values, ax=ax)
ax.set_ylabel("Кількість")
st.pyplot(fig)
st.write("Ця діаграма порівнює загальну кількість штрафних спроб (FTA) і штрафних влучань (FTM) для порівняння.")

# Візуалізація 9: Залежність GP або MIN від TARGET_5Yrs ++++++
st.subheader("9. Залежність між GP або MIN від TARGET_5Yrs")
option = st.radio("Оберіть показник", ['GP', 'MIN'])
data_grouped = Data.groupby('TARGET_5Yrs')[option].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=data_grouped['TARGET_5Yrs'], y=data_grouped[option], ax=ax)
ax.set_xlabel("Чи залишився гравець в NBA через 5 років")
ax.set_ylabel(f"Середній {option}")
st.pyplot(fig)
st.write(f"Ця діаграма демонструє середнє значення параметру {option} в залежності від того, чи залишився гравець в NBA через 5 років.")

# Візуалізація 10: Розподіл середньої кількості блокшотів (BLK) та перехоплень (STL) ++++++
st.subheader("10. Розподіл блокшотів та перехоплень")
fig, ax = plt.subplots()
sns.boxplot(data=Data[['BLK', 'STL']], ax=ax)
ax.set_ylabel("Кількість")
st.pyplot(fig)
st.write("Ця діаграма показує розподіл середньої кількості блокшотів (BLK) та перехоплень (STL) серед гравців NBA.")









st.title("📊 Результати навчання у вигляді метрик")

from pycaret.classification import load_model, interpret_model, pull, plot_model, predict_model

model_path = r"C:\Users\wojil\final_lr_model"

# Перевірка, чи існує файл
model = load_model(model_path)
predictions = predict_model(model, data=Data)

# st.write("📊 Оцінка моделі на тестових даних:")
# st.dataframe(predictions.head())

# Візуалізація 1: Матриця плутанини
st.subheader("1. Матриця плутанини")
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(predictions['TARGET_5Yrs'], predictions['prediction_label']), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Передбачене значення")
ax.set_ylabel("Реальне значення")
st.pyplot(fig)
st.write("Матриця плутанини показує, наскільки добре модель розпізнає класи.")

# Візуалізація 2: Розподіл ймовірностей передбачень
st.subheader("2. Розподіл ймовірностей передбачень")
fig, ax = plt.subplots()
sns.histplot(predictions['prediction_score'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Ймовірність позитивного класу")
ax.set_ylabel("Кількість спостережень")
st.pyplot(fig)
st.write("Ця діаграма показує, наскільки впевнена модель у своїх передбаченнях.")

# Візуалізація 3: Коефіцієнти логістичної регресії (альтернатива важливості ознак)
st.subheader("3. Вплив ознак у логістичній регресії")
feature_importance = pd.Series(model.coef_[0], index=Data.drop(columns=['TARGET_5Yrs']).columns)
fig, ax = plt.subplots()
feature_importance.sort_values().plot(kind='barh', ax=ax)
ax.set_xlabel("Вплив ознаки")
ax.set_ylabel("Ознаки")
st.pyplot(fig)
st.write("Ця діаграма показує, які ознаки мають найбільший вплив на передбачення моделі.")


# Візуалізація 4: Розподіл передбачень за реальними класами
st.subheader("4. Розподіл передбачень")
fig, ax = plt.subplots()
sns.boxplot(x=predictions['TARGET_5Yrs'], y=predictions['prediction_score'], ax=ax)
ax.set_xlabel("Реальний клас")
ax.set_ylabel("Ймовірність прогнозу")
st.pyplot(fig)
st.write("Цей графік показує, як модель оцінює ймовірності для різних класів.")

# Візуалізація 5: Точність передбачень (кругова діаграма)
st.subheader("5. Точність передбачень")
labels = ["Вірні передбачення", "Невірні передбачення"]
values = [(predictions['TARGET_5Yrs'] == predictions['prediction_label']).sum(), (predictions['TARGET_5Yrs'] != predictions['prediction_label']).sum()]
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.3f%%', colors=['lightgreen', 'red'])
st.pyplot(fig)
st.write("Ця кругова діаграма показує частку правильних та неправильних передбачень моделі.")

# Візуалізація 6: Графік залишків
st.subheader("6. Графік залишків")
fig, ax = plt.subplots()
sns.scatterplot(x=predictions.index, y=predictions['prediction_score'] - predictions['TARGET_5Yrs'], ax=ax)
ax.set_xlabel("Індекс запису")
ax.set_ylabel("Залишки")
st.pyplot(fig)
st.write("Цей графік допомагає аналізувати помилки передбачень.")

# Візуалізація 7: Гістограма помилок
st.subheader("7. Гістограма помилок")
fig, ax = plt.subplots()
sns.histplot(predictions['prediction_score'] - predictions['TARGET_5Yrs'], bins=20, kde=True, ax=ax)
ax.set_xlabel("Різниця між передбаченням і реальним значенням")
st.pyplot(fig)
st.write("Ця гістограма показує розподіл помилок моделі.")

# Візуалізація 8: Середні передбачені значення за групами
st.subheader("8. Середні передбачені значення")
fig, ax = plt.subplots()
sns.barplot(x=predictions['TARGET_5Yrs'], y=predictions['prediction_score'], ax=ax)
ax.set_xlabel("Реальний клас")
ax.set_ylabel("Середнє передбачене значення")
st.pyplot(fig)
st.write("Цей графік демонструє середні передбачення для кожного класу.")

# Візуалізація 9: Розподіл передбачень за гравцями
st.subheader("9. Розподіл передбачень за гравцями")
fig, ax = plt.subplots()
sns.boxplot(x=predictions['prediction_label'], y=predictions['PTS'], ax=ax)
ax.set_xlabel("Передбачений клас")
ax.set_ylabel("Очки за гру")
st.pyplot(fig)
st.write("Ця діаграма показує, як очки за гру співвідносяться з передбаченнями моделі.")

# Візуалізація 10: Розподіл за зіграними матчами
st.subheader("10. Розподіл за зіграними матчами")
fig, ax = plt.subplots()
sns.boxplot(x=predictions['prediction_label'], y=predictions['GP'], ax=ax)
ax.set_xlabel("Передбачений клас")
ax.set_ylabel("Кількість зіграних матчів")
st.pyplot(fig)
st.write("Ця діаграма демонструє залежність кількості зіграних матчів від передбаченого класу.")


