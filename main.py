import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import load_model, predict_model
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Кластеризація країн", layout="wide")

st.title("Лабораторна робота №2 — Кластеризація країн")

uploaded_file = st.file_uploader("Завантажте CSV-файл", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успішно завантажено!")

    st.subheader("Попередній перегляд даних")
    st.dataframe(df.head())

    st.subheader("Описова статистика")
    st.dataframe(df.describe())

    # 1. Перевірка наявності необхідних колонок
    required_columns = {'country', 'child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'}
    if not required_columns.issubset(df.columns):
        st.error("Структура завантаженого файлу не валідна!")
        st.stop()

    # 2. Динамічний опис колонок
    st.subheader("Опис ознак")
    st.dataframe(pd.DataFrame({col: df[col].dtype for col in df.columns}.items(), columns=["Ознака", "Тип даних"]))

    # 3. Візуалізація однієї з ознак
    st.subheader("Візуалізація ознак")
    selected_feature = st.selectbox("Оберіть ознаку для побудови гістограми:", df.columns[1:])
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    ax.set_title(f"Розподіл ознаки: {selected_feature}")
    ax.set_xlabel(selected_feature)
    st.pyplot(fig)

    # 4. Завантаження моделі та кластеризація
    st.subheader("Кластеризація")
    model = load_model("load_model")
    clustered_df = predict_model(model, data=df.drop(columns=['country']))
    clustered_df['country'] = df['country']

    st.success("Кластеризацію завершено успішно!")

    # 5. Вивід результатів кластеризації
    st.subheader("Країни та відповідні кластери")
    st.dataframe(clustered_df[['country', 'Cluster']])

    # 6. Аналіз кластерів
    st.subheader("Аналіз кластерів (середні значення)")
    cluster_summary = clustered_df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_summary)

    # 7. Метрики якості кластеризації
    st.subheader("Метрики якості кластеризації")
    try:
        X = df.drop(columns=['country'])
        labels = clustered_df['Cluster']

        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)

        st.markdown(f"""
        - **Silhouette:** {silhouette:.3f}  
        - **Davies-Bouldin:** {db_index:.3f}  
        - **Calinski-Harabasz:** {calinski:.3f}
        """)
    except Exception as e:
        st.warning(f"Не вдалося обчислити метрики: {e}")

    # 8. Візуалізація кількості країн по кластерам
    st.subheader("Розподіл країн по кластерам")
    fig2, ax2 = plt.subplots()
    cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax2)
    ax2.set_xlabel("Кластер")
    ax2.set_ylabel("Кількість країн")
    ax2.set_title("Кількість країн у кожному кластері")
    st.pyplot(fig2)
