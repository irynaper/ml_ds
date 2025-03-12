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
        🚀 ПЗ2. Ярошенко Ярослав. КНТу-22-1. 🚀
    </h1>
    """,
    unsafe_allow_html=True
)
# Заголовок сторінки
st.title("📊 Аналіз якості вина: Метрики та Візуалізація")

# Завантаження датасету
Data = get_data('wine')

# Додавання опису датасету
st.markdown("""
            # Опис датасету "wine.csv"

Даний датасет містить хімічні характеристики червоних і білих вин, а також їхню оцінку якості. Він використовується для аналізу факторів, що впливають на якість вина, побудови моделей машинного навчання для прогнозування оцінки якості та проведення дослідницького аналізу даних.

## Загальна інформація про датасет
- Кількість записів: **6497**  
- Кількість ознак: **13**  
- Типи змінних:  
  - **11 числових** (`float64`)  
  - **1 цілочисельна** (`int64`)  
  - **1 категоріальна** (`object`)  

## Опис ознак

### Хімічні характеристики:
- **`fixed acidity` (фіксована кислотність)** – місткість некарбонатних кислот у вині.  
  - Діапазон значень: **3.8 – 15.9**  
  - Середнє значення: **7.21**  

- **`volatile acidity` (летка кислотність)** – місткість оцтової кислоти, що впливає на смак вина.  
  - Діапазон значень: **0.08 – 1.58**  
  - Середнє значення: **0.34**  

- **`citric acid` (лимонна кислота)** – додає вину свіжості.  
  - Діапазон значень: **0.00 – 1.66**  
  - Середнє значення: **0.32**  

- **`residual sugar` (залишковий цукор)** – кількість цукру, що залишилася після бродіння.  
  - Діапазон значень: **0.6 – 65.8 г/л**  
  - Середнє значення: **5.44 г/л**  

- **`chlorides` (хлориди)** – концентрація солі у вині, що впливає на смак.  
  - Діапазон значень: **0.009 – 0.611 г/л**  
  - Середнє значення: **0.056 г/л**  

- **`free sulfur dioxide` (вільний діоксид сірки)** – місткість SO₂, що запобігає окисленню та розвитку бактерій.  
  - Діапазон значень: **1 – 289 мг/л**  
  - Середнє значення: **30.5 мг/л**  

- **`total sulfur dioxide` (загальний діоксид сірки)** – загальний вміст SO₂, що використовується як консервант.  
  - Діапазон значень: **6 – 440 мг/л**  
  - Середнє значення: **115.7 мг/л**  

- **`density` (щільність)** – щільність вина, що залежить від вмісту спирту та цукру.  
  - Діапазон значень: **0.9871 – 1.03898 г/см³**  
  - Середнє значення: **0.9947 г/см³**  

- **`pH` (кислотність)** – рівень pH вина, що впливає на його стабільність і смак.  
  - Діапазон значень: **2.72 – 4.01**  
  - Середнє значення: **3.22**  

- **`sulphates` (сульфати)** – рівень сульфатів, що впливають на аромат і смак вина.  
  - Діапазон значень: **0.22 – 2.00 г/л**  
  - Середнє значення: **0.53 г/л**  

- **`alcohol` (алкоголь)** – вміст спирту у вині (%).  
  - Діапазон значень: **8.0 – 14.9%**  
  - Середнє значення: **10.49%**  

### Цільова змінна:
- **`quality` (якість вина)** – оцінка якості за шкалою від **3 до 9** (цілочисельне значення).  

### Категоріальна ознака:
- **`type` (тип вина)** – категоріальна ознака, що позначає тип вина:  
  - `"red"` – червоне вино  
  - `"white"` – біле вино  

## Використання датасету
Цей датасет дозволяє проводити аналіз впливу різних хімічних характеристик на якість вина та використовувати ці дані для прогнозування якості на основі хімічного складу.
""")

label_encoder = LabelEncoder()
Data['type'] = label_encoder.fit_transform(Data['type'])

# Показ первых строк датасета
if st.checkbox("Показать первые строки датасета"):
    st.write(Data.head())

# Описание переменных
st.sidebar.header("Описание переменных")
variable_description = st.sidebar.selectbox("Выберите переменную для описания:", Data.columns)
if variable_description in Data.columns:
    st.sidebar.write(f"**{variable_description}**: Описание переменной...")

# Выбор переменных для анализа
st.sidebar.header("Настройки визуализации")
x_axis = st.sidebar.selectbox("Выберите переменную для оси X:", Data.columns)
y_axis = st.sidebar.selectbox("Выберите переменную для оси Y:", Data.columns)
color = st.sidebar.selectbox("Выберите переменную для цвета:", Data.columns)

# Ползунок для выбора диапазона значений
range_slider = st.sidebar.slider(
    "Выберите диапазон значений:",
    min_value=float(Data[x_axis].min()),
    max_value=float(Data[x_axis].max()),
    value=(float(Data[x_axis].min()), float(Data[x_axis].max()))
)

# Фильтрация данных по выбранному диапазону
filtered_data = Data[(Data[x_axis] >= range_slider[0]) & (Data[x_axis] <= range_slider[1])]

# Построение интерактивного графика
st.header("📈 Интерактивный график")
fig = px.scatter(filtered_data, x=x_axis, y=y_axis, color=color, title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig)

# Дополнительные интерактивные элементы
st.sidebar.header("Дополнительные настройки")
show_histogram = st.sidebar.checkbox("Показать гистограмму")

if show_histogram:
    hist_axis = st.sidebar.selectbox("Выберите переменную для гистограммы:", Data.columns)
    fig_hist = px.histogram(filtered_data, x=hist_axis, title=f"Гистограмма {hist_axis}")
    st.plotly_chart(fig_hist)

# Boxplot для выбранной переменной
if st.sidebar.checkbox("Показать Boxplot"):
    boxplot_axis = st.sidebar.selectbox("Выберите переменную для Boxplot:", Data.columns)
    fig_box = px.box(filtered_data, y=boxplot_axis, title=f"Boxplot {boxplot_axis}")
    st.plotly_chart(fig_box)

# Heatmap корреляций
if st.checkbox("Показать Heatmap корреляций"):
    st.header("Тепловая карта корреляций")
    corr = Data.corr()
    fig_heatmap = px.imshow(corr, text_auto=True, title="Тепловая карта корреляций")
    st.plotly_chart(fig_heatmap)

# 3D-график
if st.checkbox("Показать 3D-график"):
    st.header("3D-график")
    x_3d = st.selectbox("Выберите переменную для оси X (3D):", Data.columns)
    y_3d = st.selectbox("Выберите переменную для оси Y (3D):", Data.columns)
    z_3d = st.selectbox("Выберите переменную для оси Z (3D):", Data.columns)
    fig_3d = px.scatter_3d(Data, x=x_3d, y=y_3d, z=z_3d, color=color, title=f"3D-график: {x_3d}, {y_3d}, {z_3d}")
    st.plotly_chart(fig_3d)

# Загрузка модели и прогнозирование
st.header("🤖 Прогнозирование с использованием модели")
if st.checkbox("Загрузить модель и сделать прогноз"):
    model = load_model('C:/Users/Dasha/final_rf_model')  # Укажите путь к вашей модели
    st.write("Модель успешно загружена!")

    # Преобразуем категориальный столбец 'type' в числовой
    label_encoder = LabelEncoder()
    Data['type'] = label_encoder.fit_transform(Data['type'])

    # Прогнозирование
    y_test = Data['quality']
    X_test = Data.drop(columns=['quality'])
    predictions = predict_model(model, data=X_test)
    y_pred = predictions['prediction_label']

    # Метрики модели
    st.subheader("Метрики модели")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    st.write(f"**Точность (Accuracy):** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")

    # Матрица ошибок
    st.subheader("Матрица ошибок")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_conf_matrix = px.imshow(conf_matrix, text_auto=True, title="Матрица ошибок")
    st.plotly_chart(fig_conf_matrix)

# Дополнительные визуализации
st.header("📊 Дополнительные визуализации")

# Парные графики
if st.checkbox("Показать парные графики"):
    st.subheader("Парные графики")
    pair_plot_vars = st.multiselect("Выберите переменные для парных графиков:", Data.columns, default=['alcohol', 'pH', 'quality'])
    fig_pair = sns.pairplot(Data[pair_plot_vars], hue='quality' if 'quality' in pair_plot_vars else None)
    st.pyplot(fig_pair)

# KDE-график
if st.checkbox("Показать KDE-график"):
    st.subheader("KDE-график")
    kde_x = st.selectbox("Выберите переменную для KDE:", Data.columns)
    fig_kde = px.density_contour(Data, x=kde_x, title=f"KDE-график для {kde_x}")
    st.plotly_chart(fig_kde)

# Streamgraph
if st.checkbox("Показать Streamgraph"):
    st.subheader("Streamgraph")
    stream_x = st.selectbox("Выберите переменную для оси X (Streamgraph):", Data.columns)
    stream_y = st.selectbox("Выберите переменную для оси Y (Streamgraph):", Data.columns)
    fig_stream = px.area(Data, x=stream_x, y=stream_y, title=f"Streamgraph: {stream_x} vs {stream_y}")
    st.plotly_chart(fig_stream)

# 📊 1. 3D-розподіл алкоголю та кислотності
fig = px.scatter_3d(Data, x='alcohol', y='pH', z='volatile acidity', color='quality',
                     title="📊 3D-розподіл алкоголю, pH та леткої кислотності",
                     opacity=0.7)
st.plotly_chart(fig)

# 🎨 2. Heatmap з анімацією змін рівня pH (імітація тренду)
fig = px.density_heatmap(Data, x="fixed acidity", y="pH", animation_frame="quality",
                         title="🎨 Зміна рівня pH залежно від кислотності (анімований графік)",
                         color_continuous_scale="Viridis")
st.plotly_chart(fig)

# 🌀 3. Полярний графік середніх значень характеристик вина
polar_data = Data.drop(columns=["quality", "type"]).mean()
angles = list(polar_data.index) + [polar_data.index[0]]
values = list(polar_data) + [polar_data[0]]
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=values, theta=angles, fill='toself', name='Середні значення'))
fig.update_layout(title="🌀 Полярний графік середніх характеристик вина")
st.plotly_chart(fig)

# 🎭 4. Swarm plot (роєвий графік) алкоголю vs якості
fig, ax = plt.subplots(figsize=(8, 6))
sns.swarmplot(x=Data["quality"], y=Data["alcohol"], ax=ax)
ax.set_title("🎭 Swarm Plot: Алкоголь vs Якість вина")
ax.set_xlabel("Оцінка якості")
ax.set_ylabel("Алкоголь (%)")
st.pyplot(fig)

# 🔀 5. Correlation dendrogram (дендрограма кореляцій між характеристиками)
Data_numeric = Data.select_dtypes(include=[float, int])  # Оставляем только числовые колонки
corr = Data_numeric.corr()

fig, ax = plt.subplots(figsize=(10, 5))
dist = 1 - corr  # Преобразуем корреляцию в матрицу расстояний
linkage_matrix = sch.linkage(dist, method='ward')

sch.dendrogram(linkage_matrix, labels=corr.columns, leaf_rotation=90, ax=ax)
ax.set_title("🔀 Дендрограма кореляцій між характеристиками вина")
st.pyplot(fig)

# 📈 6. KDE-графік взаємозв'язку алкоголю та рівня pH
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(x=Data["alcohol"], y=Data["pH"], cmap="mako", fill=True, ax=ax)
ax.set_title("📈 KDE-графік: Алкоголь vs pH")
ax.set_xlabel("Алкоголь (%)")
ax.set_ylabel("pH")
st.pyplot(fig)

# 📌 7. Lollipop chart (льодяникова діаграма) важливості параметрів
importance = abs(Data_numeric.corr()["quality"]).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.stem(importance.index, importance.values)
ax.set_title("📌 Важливість параметрів для якості вина (Lollipop chart)")
ax.set_ylabel("Коефіцієнт кореляції з якістю")
st.pyplot(fig)

# 🎢 8. Streamgraph (графік потоків) розподілу типів вина
fig = px.area(Data.groupby(["quality", "type"]).size().reset_index(name="count"),
              x="quality", y="count", color="type", line_group="type",
              title="🎢 Streamgraph: Тип вина за якістю")
st.plotly_chart(fig)

# 🎯 9. Bullseye chart (мішень) – співвідношення алкоголю та леткої кислотності
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=Data["alcohol"], y=Data["volatile acidity"], hue=Data["quality"], s=100, edgecolor="k", ax=ax)
ax.set_title("🎯 Bullseye chart: Алкоголь vs Летка кислотність")
ax.set_xlabel("Алкоголь (%)")
ax.set_ylabel("Летка кислотність")
st.pyplot(fig)

# 🧩 10. Mosaic plot залежності типу вина та якості
Data["quality"] = Data["quality"].astype(str)  # Преобразуем в строку, если нужно
Data["type"] = Data["type"].astype(str)  # Преобразуем в строку, если нужно

fig, ax = plt.subplots(figsize=(8, 6))
mosaic(Data, ["type", "quality"], ax=ax)
ax.set_title("🧩 Mosaic Plot: Тип вина vs Якість")
st.pyplot(fig)

# Завантаження моделі через PyCaret
model = load_model('C:/Users/Dasha/final_rf_model')

# Преобразуем категориальный столбец 'type' в числовой
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

# 🔹 1. Гістограма розподілу якості вина
fig, ax = plt.subplots()
sns.histplot(Data['quality'], bins=7, kde=True, ax=ax)
ax.set_title('📊 Розподіл якості вина')
ax.set_xlabel('Оцінка якості вина')
ax.set_ylabel('Кількість зразків')
st.pyplot(fig)

# 🔹 2. Матриця помилок
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Передбачене значення')
ax.set_ylabel('Реальне значення')
ax.set_title('Матриця помилок')
st.pyplot(fig)

# 🔹 3. Взаємозв’язок алкоголю та якості вина
fig, ax = plt.subplots()
sns.scatterplot(x=Data['alcohol'], y=Data['quality'], ax=ax)
ax.set_title('Алкоголь vs Якість вина')
ax.set_xlabel('Алкоголь (%)')
ax.set_ylabel('Оцінка якості')
st.pyplot(fig)

# 🔹 4. Кореляційна матриця
fig, ax = plt.subplots()
sns.heatmap(Data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Кореляційна матриця характеристик вина')
st.pyplot(fig)

# 🔹 5. Гістограма рівня pH
fig, ax = plt.subplots()
sns.histplot(Data['pH'], bins=20, kde=True, ax=ax)
ax.set_title('Розподіл рівня pH')
ax.set_xlabel('pH')
ax.set_ylabel('Частота')
st.pyplot(fig)

# 🔹 6. Boxplot: Алкоголь vs Якість
fig, ax = plt.subplots()
sns.boxplot(x=Data['quality'], y=Data['alcohol'], ax=ax)
ax.set_title('Boxplot: Алкоголь vs Якість вина')
ax.set_xlabel('Оцінка якості')
ax.set_ylabel('Алкоголь (%)')
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
st.pyplot(sns.pairplot(Data[['alcohol', 'pH', 'volatile acidity', 'residual sugar', 'quality']], hue='quality'))

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
ax.set_title('Boxplot: Кислотність vs Якість вина')
ax.set_xlabel('Оцінка якості')
ax.set_ylabel('Фіксована кислотність')
st.pyplot(fig)

