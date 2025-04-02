import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.datasets import get_data

# Завантаження датасету glass
df = get_data('glass')

# Налаштування параметрів відображення Pandas
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 2500)
pd.set_option('display.max_colwidth', 2500)

# Вивід перших 5 рядків датасету
print("Перші 5 рядків датасету:")
print(df.head())

# Опис статистики числових ознак
print("\nСтатистичний опис:")
print(df.describe())

# Перевірка на відсутні значення
print("\nКількість відсутніх значень:")
print(df.isnull().sum())

# Візуалізація розподілу цільової змінної (Type)
plt.figure(figsize=(6,4))
sns.countplot(x='Type', data=df)
plt.title("Розподіл типів скла")
plt.xlabel("Тип скла")
plt.ylabel("Кількість зразків")
plt.show()

# Побудова гістограм для числових ознак (крім Type)
num_features = df.select_dtypes(include=['float64', 'int64']).columns.drop('Type')
df[num_features].hist(bins=15, figsize=(15, 10))
plt.suptitle("Гістограми числових ознак")
plt.show()

# Кореляційна матриця числових ознак
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Кореляційна матриця ознак")
plt.show()

from pycaret.classification import *

target = "Type"

# Налаштовуємо середовище PyCaret (session_id встановлено для відтворюваності)
clf = setup(data=df, target=target, session_id=123, verbose=False)

# Порівнюємо моделі
best_model = compare_models()

# Збереження найкращої моделі разом з трансформаційним пайплайном
save_model(best_model, "glass_best_model")

# Вивід інформації про найкращу модель
print("Найкраща модель:")
print(best_model)