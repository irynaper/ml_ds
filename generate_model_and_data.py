
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, save_model
import pandas as pd

# Загружаем датасет
print("Завантаження датасету...")
nba = get_data('nba')
nba.dropna(inplace=True)
nba.to_csv('nba.csv', index=False)
print("✅ Файл nba.csv збережено.")

# Настройка и обучение модели
print("Навчання моделі...")
clf = setup(data=nba, target='Pos', session_id=123, silent=True, verbose=False)
best_model = compare_models()

# Сохранение модели
save_model(best_model, 'nba_classifier_model')
print("✅ Модель збережено як nba_classifier_model.pkl")
