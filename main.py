import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import load_model, predict_model

st.set_page_config(page_title="Кластеризація країн", layout="wide")

st.title("Лабораторна робота №2 — Кластеризація країн")

uploaded_file = st.file_uploader("Файл", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успішно завантажено!")

    st.subheader("Попередній перегляд даних")
    st.dataframe(df.head())

    # 3. Опис ознак
    st.subheader("Опис ознак (features)")
    features = {
        'country': 'Назва країни.',
        'child_mort': 'Дитяча смертність (на 1000 дітей).',
        'exports': 'Експорт як % від ВВП.',
        'health': 'Витрати на охорону здоров’я (% від ВВП).',
        'imports': 'Імпорт як % від ВВП.',
        'income': 'Середній дохід на душу населення (USD).',
        'inflation': 'Інфляція (%).',
        'life_expec': 'Очікувана тривалість життя (роки).',
        'total_fer': 'Коефіцієнт народжуваності.',
        'gdpp': 'ВВП на душу населення (USD).'
    }
    st.table(pd.DataFrame(features.items(), columns=["Ознака", "Опис"]))

    # 4. Інтерактивна візуалізація
    st.subheader("Візуалізація даних")
    selected_feature = st.selectbox("Оберіть ознаку для побудови гістограми:", df.columns[1:])
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    ax.set_title(f"Розподіл ознаки: {selected_feature}")
    ax.set_xlabel(selected_feature)
    st.pyplot(fig)

    # 5. Завантаження моделі та передбачення
    st.subheader("Завантаження моделі та кластеризація")
    model = load_model("load_model")  # важливо: файл має бути в одній папці з lab2.py
    clustered_df = predict_model(model, data=df.drop(columns=['country']))
    clustered_df['country'] = df['country']

    st.success("Кластеризацію завершено успішно!")

    # 6. Результати кластеризації
    st.subheader("Країни та відповідні кластери")
    st.dataframe(clustered_df[['country', 'Cluster']])

    # 7. Аналіз кластерів
    st.subheader("Аналіз кластерів (середні значення)")
    cluster_summary = clustered_df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_summary)

    # 8. Візуалізація розподілу країн по кластерам
    st.subheader("🗺Кількість країн у кожному кластері")
    fig2, ax2 = plt.subplots()
    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax2)
    ax2.set_xlabel("Кластер")
    ax2.set_ylabel("Кількість країн")
    ax2.set_title("Розподіл країн по кластерам")
    st.pyplot(fig2)
