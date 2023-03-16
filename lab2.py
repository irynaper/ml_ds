# Імпортуеємо бібліотеку для імпорту даних
from pycaret.datasets import get_data

import streamlit as st

st.subheader("Початкові дані датасету income")

# Імпортуємо наш датасет
dataset = get_data('income')
# Виводимо його форму, сам датасет та наявні поля
dataset


# Переводимо поле з булевим типом зі значень 0 та 1 у True та False
dataset["income >50K"] = [i == 1 for i in dataset["income >50K"]]
# Додаємо поле capital-delta замість полів capital-gain та capital-loss
dataset["capital-delta"] = dataset["capital-gain"] - dataset["capital-loss"]

st.subheader("Переформатовані дані датасету")
st.write("Переформування датасету 'income' полягає у виделенні полів capital-gain та capital-loss і додання замість них поля capital-delta. Також формат даних поля 'income >50K' був переформатований з можливих даних 0 та 1 у True та False")
dataset.drop(labels=["capital-gain","capital-loss"], inplace=True, axis="columns")

# Ще раз виводимо датасет
dataset

# Імпортуємо бібліотеку для кластерингу 
from pycaret.clustering import *
# Ініціюємо нашу модель для датасету та зазначаємо параметр нормалізації у True та session_id у 123 задля відтворюваності викликів наступних функцій
cl = setup(dataset, normalize = True, session_id = 123)
# st.table(cl)

st.subheader("Оберіть кількість кластерів")
num_clusters = st.slider(label="Кількість кластерів", min_value=3, max_value=10, value=6,step=1)

# Обираємо метод найближчих k сусідів та кількість кластерів=4 для нашої моделі
kmeans = create_model('kmeans', num_clusters = num_clusters)

# Отримуємо наш результат
kmean_results = assign_model(kmeans)

# Виводимо результат кластерізації
kmean_results

# Виводимо графік для даної моделі
plot_model(kmeans, display_format='streamlit')

# Виводимо графік розподілу для нашої моделі
a = plot_model(kmeans, plot = 'distribution', display_format='streamlit')

st.write(type(a))

# Виводимо графік розподілу за параметром віку
plot_model(kmeans, plot = 'distribution', feature="age", display_format='streamlit')


# Виводимо графік розподілу за параметром статі
plot_model(kmeans, plot = 'distribution', feature="sex", display_format='streamlit')

# Виводимо графік розподілу за параметром відпрацьованих годин на тиждень
plot_model(kmeans, plot = 'distribution', feature="hours-per-week", display_format='streamlit')

# Виводимо графік розподілу за параметром раси
plot_model(kmeans, plot = 'distribution', feature="race", display_format='streamlit')

# Виводимо графік розподілу за параметром різниці доходу та розходу капіталу
plot_model(kmeans, plot = 'distribution', feature="capital-delta", display_format='streamlit')

# Нижче для кожного кластеру виводимо: середній вік, медіанний вік, середню дельту капіталу, медіанну дельту капіталу
for i in range(num_clusters):
    cluster = kmean_results[kmean_results["Cluster"] == f"Cluster {i}"]
    st.subheader(f"Cluster {i}")
    st.write(f"  Average age {cluster['age'].mean()}")
    st.write(f"  Median age {cluster['age'].median()}")
    st.write(f"  Average capital delta {cluster['capital-delta'].mean()}")
    st.write(f"  Median capital delta {cluster['capital-delta'].median()}")