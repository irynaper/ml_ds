import pandas as pd
import numpy as np
import streamlit as st

st.subheader("Початковий датасет")

# Імпортуеємо бібліотеку для імпорту даних
from pycaret.datasets import get_data

# Імпортуємо датасет відповідно до заданого варіанту
df = get_data('automobile')


# Перетворюємо певні категоріальні поля, які мають бути числовими, у числові
df['horsepower'] = pd.to_numeric(df['horsepower'], errors="coerce")
df['bore'] = pd.to_numeric(df['bore'], errors="coerce")
df['stroke'] = pd.to_numeric(df['stroke'], errors="coerce")
df['peak-rpm'] = pd.to_numeric(df['peak-rpm'], errors="coerce")

from pycaret.clustering import *
# Ініціюємо модель з попередньою нормалізацією  даних
cl = setup(df, normalize=True, session_id=42)
cl


#
# st.subheader("Оберіть кількість кластерів")
# num_clusters = st.slider(label="Кількість кластерів", min_value=3, max_value=10, value=6,step=1)

st.subheader("Створюємо модель...")
# Візьмемо метод к найближчих сусідів та кількість кластерів - 2
kmeans = create_model('kmeans', num_clusters=2)

# Результат кластерізації
kmean_results = assign_model(kmeans)
kmean_results
st.subheader("Графік кластерів")
# Графік кластерів
plot_model(kmeans, display_format='streamlit')
st.subheader("Графік розподілу кількості записів у кожному кластері")
# Графік розподілу кількості записів у кожному кластері
plot_model(kmeans, plot='distribution', display_format='streamlit')


# Нижче для всього датасету та для кожного кластеру виводимо середні значення для всіх числових даних
cluster_1 = kmean_results[kmean_results["Cluster"] == f"Cluster {0}"]
cluster_2 = kmean_results[kmean_results["Cluster"] == f"Cluster {1}"]
st.subheader("Оберіть числові дані, для яких ви хочете побачити середнє значення")
selected_columns = st.multiselect(" ",
               df.select_dtypes(include=np.number).columns.tolist())
for feature in selected_columns:
    st.write(f"  Mean {feature} for dataset: {df[feature].mean()}")
    st.write(f"  Mean {feature} for cluster 0: {cluster_1[feature].mean()}")
    st.write(f"  Mean {feature} for cluster 1: {cluster_2[feature].mean()}\n")
    st.write(" ")


# Виводимо графік розподілу за кількістю дверей у авто
plot_model(kmeans, plot='distribution', feature="num-of-doors", display_format='streamlit')



# Графік розподілу за системою пального
st.subheader("Графік розподілу за системою пального")
plot_model(kmeans, plot='distribution', feature="fuel-system", display_format='streamlit')


# Графік розподілу за типом двигуна
st.subheader("Графік розподілу за типом двигуна")
plot_model(kmeans, plot='distribution', feature="engine-type", display_format='streamlit')

# Графік розподілу за кількістю циліндрів двигуна
st.subheader("Графік розподілу за кількістю циліндрів двигуна")
plot_model(kmeans, plot='distribution', feature="num-of-cylinders", display_format='streamlit')

# Графік розподілу за типом кузова
st.subheader("Графік розподілу за типом кузова")
plot_model(kmeans, plot='distribution', feature="body-style", display_format='streamlit')

# Графік розподілу за типом ведучих коліс
st.subheader("Графік розподілу за типом ведучих коліс")
plot_model(kmeans, plot='distribution', feature="drive-wheels", display_format='streamlit')

# Зберігаємо модель
save_model(kmeans, model_name='kmeans_model')

conclusion = """"
Опираючись на усі дані, що були виведені вище, можемо зазначити, що дана кластерізація вийшла досить збалансованою. Враховуючи те, що даний датасет складався тільки з 201 записів ми отримали приблизне співвідношення 2:1 у кількості екземплярів кожного кластеру. Слід зазначити, що для більшої кількості кластерів, через розмір нашої вибірки, кластери стають досить малими, і через це втрачається практична цінність результатів, тому кількість кластерів - 2.

Якщо подивитися на такі параметри як розмір авто (length, width, height), масу, об'єм двигуна, ціну та інші параметри, можемо з упевненіст сказати, що в нульовому кластері ми отримали більш великі, потужні та дорожчі автомобілі ніж у першому. Це відображається не тільки в числових даних, а й в категоріальних, адже, як можна побачити на графіках, нульовий кластер зазвичай має більшу кількість ціліндрів двигуна, зазвичай має задній привід, має тільки новітні системи подачі пального та інше.

"""

st.subheader(conclusion)