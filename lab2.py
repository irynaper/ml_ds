import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження даних
try:
    from pycaret.datasets import get_data
    data = get_data('glass')
except Exception as e:
    st.error("Не вдалося завантажити дані через PyCaret. Спробуйте завантажити дані з файлу.")
    st.error(e)

# Заголовки додатку
st.title("Аналіз датасету 'glass'")
st.header("Опис ознак")

# Опис ознак
feature_description = {
    "RI": "Показник заломлення світла",
    "Na": "Вміст натрію в скла",
    "Mg": "Вміст магнію",
    "Al": "Вміст алюмінію",
    "Si": "Вміст кремнію",
    "K": "Вміст калію",
    "Ca": "Вміст кальцію",
    "Ba": "Вміст барію",
    "Fe": "Вміст заліза"
}
st.table(pd.DataFrame(feature_description.items(), columns=["Ознака", "Опис"]))

st.subheader("Перші 5 рядків даних")
st.dataframe(data.head())

# Візуалізація даних
st.header("Візуалізація даних")
feature = st.selectbox("Оберіть ознаку для візуалізації", data.columns)

# Гістограма
fig1, ax1 = plt.subplots()
ax1.hist(data[feature], bins=20, edgecolor='black')
ax1.set_title(f"Гістограма для {feature}")
ax1.set_xlabel(feature)
ax1.set_ylabel("Частота")
st.pyplot(fig1)

# Боксплот
fig2, ax2 = plt.subplots()
ax2.boxplot(data[feature])
ax2.set_title(f"Боксплот для {feature}")
ax2.set_ylabel(feature)
st.pyplot(fig2)

# Порівняння ознак через кореляційну матрицю
st.header("Порівняння ознак")
selected_features = st.multiselect("Оберіть ознаки для побудови діаграми", list(data.columns),
                                   default=list(data.columns)[:3])
if len(selected_features) >= 2:
    st.write("Побудова кореляційної матриці:")
    corr = data[selected_features].corr()
    st.dataframe(corr)
else:
    st.write("Оберіть мінімум 2 ознаки для порівняння.")

# Завантаження моделі та обчислення метрик
st.header("Результати навчання моделі")
try:
    from pycaret.classification import load_model, predict_model
    model = load_model("glass_best_model")
    st.success("Модель успішно завантажена за допомогою PyCaret!")

    # Прогнозування на даних (тут використовується весь датасет; для оцінки краще використовувати тестову вибірку)
    predictions = predict_model(model, data=data)

    # Обчислення метрик за допомогою sklearn
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

    true_labels = data['Type']
    pred_labels = predictions['Label']

    acc = accuracy_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels, average='weighted')
    prec = precision_score(true_labels, pred_labels, average='weighted')
    f1_val = f1_score(true_labels, pred_labels, average='weighted')
    report = classification_report(true_labels, pred_labels)

    st.write("**Accuracy:**", acc)
    st.write("**Recall:**", rec)
    st.write("**Precision:**", prec)
    st.write("**F1 Score:**", f1_val)

    st.subheader("Classification Report")
    st.text(report)

    # Побудова матриці плутанини
    cm = confusion_matrix(true_labels, pred_labels)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

except Exception as e:
    st.error("Не вдалося завантажити модель за допомогою load_model. Переконайтеся, що файл glass_best_model.pkl існує та збережений у правильному форматі.")
    st.error(e)
