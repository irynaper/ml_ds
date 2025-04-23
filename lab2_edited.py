import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
        🚀 ПЗ2. Ковбаса Олександр КНТ-21-3 🚀
    </h1>
    """,
    unsafe_allow_html=True
)
# Заголовок сторінки
st.title("📊 Аналіз якості соку: Метрики та Візуалізація")

# Завантаження датасету
Data = get_data('juice')

# Додавання опису датасету
st.markdown("""
            # Опис датасету "juice.csv"

Даний датасет містить хімічні характеристики червоних і білих вин, а також їхню оцінку якості. Він використовується для аналізу факторів, що впливають на якість соку, побудови моделей машинного навчання для прогнозування оцінки якості та проведення дослідницького аналізу даних.

## Загальна інформація про датасет
- Кількість записів: **6497**  
- Кількість ознак: **13**  
- Типи змінних:  
  - **11 числових** (`float64`)  
  - **1 цілочисельна** (`int64`)  
  - **1 категоріальна** (`object`)  

## Опис ознак

### Хімічні характеристики:
- **`fixed acidity` (фіксована кислотність)** – місткість некарбонатних кислот у соку.  
  - Діапазон значень: **3.8 – 15.9**  
  - Середнє значення: **7.21**  

- **`volatile acidity` (летка кислотність)** – місткість оцтової кислоти, що впливає на смак соку.  
  - Діапазон значень: **0.08 – 1.58**  
  - Середнє значення: **0.34**  

- **`citric acid` (лимонна кислота)** – додає вину свіжості.  
  - Діапазон значень: **0.00 – 1.66**  
  - Середнє значення: **0.32**  

- **`residual sugar` (залишковий цукор)** – кількість цукру, що залишилася після бродіння.  
  - Діапазон значень: **0.6 – 65.8 г/л**  
  - Середнє значення: **5.44 г/л**  

- **`chlorides` (хлориди)** – концентрація солі у соку, що впливає на смак.  
  - Діапазон значень: **0.009 – 0.611 г/л**  
  - Середнє значення: **0.056 г/л**  

- **`free sulfur dioxide` (вільний діоксид сірки)** – місткість SO₂, що запобігає окисленню та розвитку бактерій.  
  - Діапазон значень: **1 – 289 мг/л**  
  - Середнє значення: **30.5 мг/л**  

- **`total sulfur dioxide` (загальний діоксид сірки)** – загальний вміст SO₂, що використовується як консервант.  
  - Діапазон значень: **6 – 440 мг/л**  
  - Середнє значення: **115.7 мг/л**  

- **`density` (щільність)** – щільність соку, що залежить від вмісту спирту та цукру.  
  - Діапазон значень: **0.9871 – 1.03898 г/см³**  
  - Середнє значення: **0.9947 г/см³**  

- **`pH` (кислотність)** – рівень pH соку, що впливає на його стабільність і смак.  
  - Діапазон значень: **2.72 – 4.01**  
  - Середнє значення: **3.22**  

- **`sulphates` (сульфати)** – рівень сульфатів, що впливають на аромат і смак соку.  
  - Діапазон значень: **0.22 – 2.00 г/л**  
  - Середнє значення: **0.53 г/л**  

- **`juice` (Сік)** – вміст спирту у соку (%).  
  - Діапазон значень: **8.0 – 14.9%**  
  - Середнє значення: **10.49%**  

### Цільова змінна:
- **`quality` (якість соку)** – оцінка якості за шкалою від **3 до 9** (цілочисельне значення).  

### Категоріальна ознака:
- **`type` (тип соку)** – категоріальна ознака, що позначає тип соку:  
  - `"red"` – червоне вино  
  - `"white"` – біле вино  

## Використання датасету
Цей датасет дозволяє проводити аналіз впливу різних хімічних характеристик на якість соку та використовувати ці дані для прогнозування якості на основі хімічного складу.
""")

label_encoder = LabelEncoder()
Data['type'] = label_encoder.fit_transform(Data['type'])

# Показ перших рядків датасету
if st.checkbox("Показати перші рядки датасету"):
    st.write(Data.head())

# Опис змінних
st.sidebar.header("Опис змінних")
variable_description = st.sidebar.selectbox("Оберіть змінну для опису:", Data.columns)
if variable_description in Data.columns:
    st.sidebar.write(f"**{variable_description}**: Опис змінної...")

# Вибір змінних для аналізу
st.sidebar.header("Налаштування візуалізації")
x_axis = st.sidebar.selectbox("Оберіть змінну для осі X:", Data.columns)
y_axis = st.sidebar.selectbox("Оберіть змінну для осі Y:", Data.columns)
color = st.sidebar.selectbox("Оберіть змінну для кольору:", Data.columns)

# Повзунок для вибору діапазону значень
range_slider = st.sidebar.slider(
    "Оберіть діапазон значень:",
    min_value=float(Data[x_axis].min()),
    max_value=float(Data[x_axis].max()),
    value=(float(Data[x_axis].min()), float(Data[x_axis].max()))
)

# Фільтрація даних за обраним діапазоном
filtered_data = Data[(Data[x_axis] >= range_slider[0]) & (Data[x_axis] <= range_slider[1])]

# Побудова інтерактивного графіка
st.header("📈 Інтерактивний графік")
fig = px.scatter(filtered_data, x=x_axis, y=y_axis, color=color, title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig)

# Додаткові інтерактивні елементи
st.sidebar.header("Додаткові налаштування")
show_histogram = st.sidebar.checkbox("Показати гістограму")

if show_histogram:
    hist_axis = st.sidebar.selectbox("Оберіть змінну для гістограми:", Data.columns)
    fig_hist = px.histogram(filtered_data, x=hist_axis, title=f"Гістограма {hist_axis}")
    st.plotly_chart(fig_hist)

# Boxplot для обраної змінної
if st.sidebar.checkbox("Показати Boxplot"):
    boxplot_axis = st.sidebar.selectbox("Оберіть змінну для Boxplot:", Data.columns)
    fig_box = px.box(filtered_data, y=boxplot_axis, title=f"Boxplot {boxplot_axis}")
    st.plotly_chart(fig_box)

# Heatmap кореляцій
if st.checkbox("Показати Heatmap кореляцій"):
    st.header("Теплова карта кореляцій")
    corr = Data.corr()
    fig_heatmap = px.imshow(corr, text_auto=True, title="Теплова карта кореляцій")
    st.plotly_chart(fig_heatmap)

# 3D-графік
if st.checkbox("Показати 3D-графік"):
    st.header("3D-графік")
    x_3d = st.selectbox("Оберіть змінну для осі X (3D):", Data.columns)
    y_3d = st.selectbox("Оберіть змінну для осі Y (3D):", Data.columns)
    z_3d = st.selectbox("Оберіть змінну для осі Z (3D):", Data.columns)
    fig_3d = px.scatter_3d(Data, x=x_3d, y=y_3d, z=z_3d, color=color, title=f"3D-графік: {x_3d}, {y_3d}, {z_3d}")
    st.plotly_chart(fig_3d)

# Завантаження моделі та прогнозування
st.header("🤖 Прогнозування з використанням моделі")
if st.checkbox("Завантажити модель і зробити прогноз"):
    model = load_model('C:/Users/Dasha/final_rf_model')  # Вкажіть шлях до вашої моделі
    st.write("Модель успішно завантажена!")

    # Перетворюємо категоріальний стовпець 'type' у числовий
    label_encoder = LabelEncoder()
    Data['type'] = label_encoder.fit_transform(Data['type'])

    # Прогнозування
    y_test = Data['quality']
    X_test = Data.drop(columns=['quality'])
    predictions = predict_model(model, data=X_test)
    y_pred = predictions['prediction_label']

    # Метрики моделі
    st.subheader("Метрики моделі")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    st.write(f"**Точність (Accuracy):** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")

    # Матриця помилок
    st.subheader("Матриця помилок")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_conf_matrix = px.imshow(conf_matrix, text_auto=True, title="Матриця помилок")
    st.plotly_chart(fig_conf_matrix)

# Додаткові візуалізації
st.header("📊 Додаткові візуалізації")

# Парні графіки
if st.checkbox("Показати парні графіки"):
    st.subheader("Парні графіки")
    pair_plot_vars = st.multiselect("Виберіть змінні для парних графіків:", Data.columns, default=['juice', 'pH', 'quality'])
    fig_pair = sns.pairplot(Data[pair_plot_vars], hue='quality' if 'quality' in pair_plot_vars else None)
    st.pyplot(fig_pair)

# KDE-графік
if st.checkbox("Показати KDE-графік"):
    st.subheader("KDE-графік")
    kde_x = st.selectbox("Виберіть змінну для KDE:", Data.columns)
    fig_kde = px.density_contour(Data, x=kde_x, title=f"KDE-графік для {kde_x}")
    st.plotly_chart(fig_kde)

# Streamgraph
if st.checkbox("Показати Streamgraph"):
    st.subheader("Streamgraph")
    stream_x = st.selectbox("Виберіть змінну для осі X (Streamgraph):", Data.columns)
    stream_y = st.selectbox("Виберіть змінну для осі Y (Streamgraph):", Data.columns)
    fig_stream = px.area(Data, x=stream_x, y=stream_y, title=f"Streamgraph: {stream_x} vs {stream_y}")
    st.plotly_chart(fig_stream)

# 📊 1. 3D-розподіл алкоголю та кислотності
fig = px.scatter_3d(Data, x='juice', y='pH', z='volatile acidity', color='quality',
                     title="📊 3D-розподіл соку, pH та леткої кислотності",
                     opacity=0.7)
st.plotly_chart(fig)

# 🎨 2. Heatmap з анімацією змін рівня pH (імітація тренду)
fig = px.density_heatmap(Data, x="fixed acidity", y="pH", animation_frame="quality",
                         title="🎨 Зміна рівня pH залежно від кислотності (анімований графік)",
                         color_continuous_scale="Viridis")
st.plotly_chart(fig)

# 🌀 3. Полярний графік середніх значень характеристик соку
polar_data = Data.drop(columns=["quality", "type"]).mean()
angles = list(polar_data.index) + [polar_data.index[0]]
values = list(polar_data) + [polar_data[0]]
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=values, theta=angles, fill='toself', name='Середні значення'))
fig.update_layout(title="🌀 Полярний графік середніх характеристик сока")
st.plotly_chart(fig)

# 🎭 4. Swarm plot (роєвий графік) алкоголю vs якості
fig, ax = plt.subplots(figsize=(8, 6))
sns.swarmplot(x=Data["quality"], y=Data["juice"], ax=ax)
ax.set_title("🎭 Swarm Plot: Сік vs Якість сока")
ax.set_xlabel("Оцінка якості")
ax.set_ylabel("Сык (%)")
st.pyplot(fig)

# 🔀 5. Correlation dendrogram (дендрограма кореляцій між характеристиками)
Data_numeric = Data.select_dtypes(include=[float, int])  
corr = Data_numeric.corr()

fig, ax = plt.subplots(figsize=(10, 5))
dist = 1 - corr  
linkage_matrix = sch.linkage(dist, method='ward')

sch.dendrogram(linkage_matrix, labels=corr.columns, leaf_rotation=90, ax=ax)
ax.set_title("🔀 Дендрограма кореляцій між характеристиками соку")
st.pyplot(fig)

# 📈 6. KDE-графік взаємозв'язку алкоголю та рівня pH
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(x=Data["juice"], y=Data["pH"], cmap="mako", fill=True, ax=ax)
ax.set_title("📈 KDE-графік: Сік vs pH")
ax.set_xlabel("Сік (%)")
ax.set_ylabel("pH")
st.pyplot(fig)

# 📌 7. Lollipop chart (льодяникова діаграма) важливості параметрів
importance = abs(Data_numeric.corr()["quality"]).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.stem(importance.index, importance.values)
ax.set_title("📌 Важливість параметрів для якості соку (Lollipop chart)")
ax.set_ylabel("Коефіцієнт кореляції з якістю")
st.pyplot(fig)

# 🎢 8. Streamgraph (графік потоків) розподілу типів соку
fig = px.area(Data.groupby(["quality", "type"]).size().reset_index(name="count"),
              x="quality", y="count", color="type", line_group="type",
              title="🎢 Streamgraph: Тип соку за якістю")
st.plotly_chart(fig)

# 🎯 9. Bullseye chart (мішень) – співвідношення алкоголю та леткої кислотності
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=Data["juice"], y=Data["volatile acidity"], hue=Data["quality"], s=100, edgecolor="k", ax=ax)
ax.set_title("🎯 Bullseye chart: Сік vs Летка кислотність")
ax.set_xlabel("Сік (%)")
ax.set_ylabel("Летка кислотність")
st.pyplot(fig)

# 🧩 10. Mosaic plot залежності типу соку та якості
Data["quality"] = Data["quality"].astype(str) 
Data["type"] = Data["type"].astype(str)  

fig, ax = plt.subplots(figsize=(8, 6))
mosaic(Data, ["type", "quality"], ax=ax)
ax.set_title("🧩 Mosaic Plot: Тип соку vs Якість")
st.pyplot(fig)

# Завантаження моделі через PyCaret
model = load_model('C:/Users/Dasha/final_rf_model')

label_encoder = LabelEncoder()
Data['type'] = label_encoder.fit_transform(Data['type'])

# Використання моделі для прогнозування
y_test = Data['quality']
X_test = Data.drop(columns=['quality'])
predictions = predict_model(model, data=X_test)
y_pred = predictions['prediction_label']

# 📌 Метрики моделі
st.header("📌 Метрики моделі")

y_test = y_test.astype(float)
y_pred = y_pred.astype(float)

# 🔹 1. Точність моделі (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### 🔹 Точність моделі (Accuracy): **{accuracy:.4f}**")

# 🔹 2. Precision, Recall, F1-score
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

st.write(f"### 🔹 Precision: **{precision:.4f}**")
st.write(f"### 🔹 Recall: **{recall:.4f}**")
st.write(f"### 🔹 F1-score: **{f1:.4f}**")

# 🔹 3. Повний звіт класифікації
st.text("### 📋 Classification Report:")
st.text(classification_report(y_test, y_pred, zero_division=0))

# 📊 Візуалізація
st.header("📊 Візуалізація")

# 🔹 1. Гістограма розподілу якості соку
fig, ax = plt.subplots()
sns.histplot(Data['quality'], bins=7, kde=True, ax=ax)
ax.set_title('📊 Розподіл якості соку')
ax.set_xlabel('Оцінка якості соку')
ax.set_ylabel('Кількість зразків')
st.pyplot(fig)

# 🔹 2. Матриця помилок
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Передбачене значення')
ax.set_ylabel('Реальне значення')
ax.set_title('Матриця помилок')
st.pyplot(fig)

# 🔹 3. Взаємозв’язок алкоголю та якості соку
fig, ax = plt.subplots()
sns.scatterplot(x=Data['juice'], y=Data['quality'], ax=ax)
ax.set_title('Сік vs Якість соку')
ax.set_xlabel('Сік (%)')
ax.set_ylabel('Оцінка якості')
st.pyplot(fig)

# 🔹 4. Кореляційна матриця
fig, ax = plt.subplots()
sns.heatmap(Data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Кореляційна матриця характеристик соку')
st.pyplot(fig)

# 🔹 5. Гістограма рівня pH
fig, ax = plt.subplots()
sns.histplot(Data['pH'], bins=20, kde=True, ax=ax)
ax.set_title('Розподіл рівня pH')
ax.set_xlabel('pH')
ax.set_ylabel('Частота')
st.pyplot(fig)

# 🔹 6. Boxplot: Сік vs Якість
fig, ax = plt.subplots()
sns.boxplot(x=Data['quality'], y=Data['juice'], ax=ax)
ax.set_title('Boxplot: Сік vs Якість соку')
ax.set_xlabel('Оцінка якості')
ax.set_ylabel('Сік (%)')
st.pyplot(fig)

# 🔹 7. Гістограма залишкового цукру
fig, ax = plt.subplots()
sns.histplot(Data['residual sugar'], bins=30, kde=True, ax=ax)
ax.set_title('Розподіл залишкового цукру')
ax.set_xlabel('Залишковий цукор (г/л)')
ax.set_ylabel('Частота')
st.pyplot(fig)

# 🔹 8. Гістограма леткої кислотності
fig, ax = plt.subplots()
sns.histplot(Data['volatile acidity'], bins=30, kde=True, ax=ax)
ax.set_title('Розподіл леткої кислотності')
ax.set_xlabel('Летка кислотність')
ax.set_ylabel('Частота')
st.pyplot(fig)

# 🔹 9. Парні графіки для основних характеристик
st.write("### Парні графіки для основних характеристик")
st.pyplot(sns.pairplot(Data[['juice', 'pH', 'volatile acidity', 'residual sugar', 'quality']], hue='quality'))

# 🔹 10. Гістограма сульфатів
fig, ax = plt.subplots()
sns.histplot(Data['sulphates'], bins=30, kde=True, ax=ax)
ax.set_title('Розподіл сульфатів')
ax.set_xlabel('Сульфати')
ax.set_ylabel('Частота')
st.pyplot(fig)

# 🔹 11. Boxplot: Кислотність vs Якість
fig, ax = plt.subplots()
sns.boxplot(x=Data['quality'], y=Data['fixed acidity'], ax=ax)
ax.set_title('Boxplot: Кислотність vs Якість соку')
ax.set_xlabel('Оцінка якості')
ax.set_ylabel('Фіксована кислотність')
st.pyplot(fig)

