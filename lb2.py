import streamlit as st
import plotly.express as px
from pycaret.classification import load_model, setup, plot_model
from pycaret.datasets import get_data

st.title("Лабораторна робота №2 «Робота з візуалізацією даних та Streamlit»")

st.header("1. Завантаження та огляд даних")
dataset = get_data('telescope')
st.success("Дані було успішно завантажено з PyCaret.")
st.dataframe(dataset.head())

st.subheader("Опис ознак")
st.markdown("""
- **alpha** – параметр, що характеризує напрямок спостереження.  
- **delta** – кутова координата об'єкта.  
- **width** – ширина об'єкта у пікселях.  
- **length** – довжина об'єкта у пікселях.  
- **size** – розмір об'єкта.  
- **conc** – коефіцієнт концентрації світла.  
- **conc1** – додатковий параметр концентрації.  
- **asym** – асиметрія об'єкта.  
- **m3long** – третій момент розподілу довжини.  
- **m3trans** – третій момент розподілу ширини.  
- **fAlpha** – фіксований параметр α.  
- **fLength** – фіксована довжина.  
- **Class** – цільова змінна (гамма-промені або космічні промені).  
""")

st.header("2. Візуалізація даних")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Гістограми", "Boxplot", "Кореляційна матриця", "Scatter Matrix", "Розподіл Class"])

with tab1:
    st.subheader("Гістограма для обраної ознаки")
    numeric_cols = dataset.select_dtypes(include=["float64", "int64"]).columns.tolist()
    selected_col = st.selectbox("Оберіть колонку для гістограми", numeric_cols)
    fig = px.histogram(dataset, x=selected_col, nbins=30, title=f"Гістограма для {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Boxplot для обраної ознаки")
    selected_col = st.selectbox("Оберіть колонку для Boxplot", numeric_cols)
    fig = px.box(dataset, y=selected_col, title=f"Boxplot для {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Кореляційна матриця")
    corr = dataset[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Кореляційна матриця")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Scatter Matrix")
    selected_cols = st.multiselect("Оберіть 3+ ознаки для Scatter Matrix", numeric_cols, default=numeric_cols[:3])
    if len(selected_cols) >= 3:
        fig = px.scatter_matrix(dataset, dimensions=selected_cols, color="Class",
                                title="Матриця розсіювання числових ознак")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Виберіть принаймні 3 ознаки.")

with tab5:
    st.subheader("Розподіл цільової змінної 'Class'")
    counts = dataset["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Кількість"]
    fig = px.bar(counts, x="Class", y="Кількість", title="Розподіл цільової змінної")
    st.plotly_chart(fig, use_container_width=True)

st.header("3. Завантаження навченої моделі та оцінка")
try:
    setup(data=dataset, target="Class", session_id=42)

    model = load_model("telescope_model")
    st.success("Модель було успішно завантажено.")

    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    score = model.score(X, y)
    st.write(f"**Оцінка точності моделі:** {score:.2f}")

    st.subheader("Графічне відображення результатів навчання")
    eval_option = st.selectbox("Оберіть графік для відображення", 
                                ["Confusion Matrix", "ROC Curve", "Feature Importance", "Precision-Recall Curve"])

    if eval_option == "Confusion Matrix":
        plot_model(model, plot='confusion_matrix', display_format='streamlit')
    elif eval_option == "ROC Curve":
        plot_model(model, plot='auc', display_format='streamlit')
    elif eval_option == "Feature Importance":
        plot_model(model, plot='feature', display_format='streamlit')
    elif eval_option == "Precision-Recall Curve":
        plot_model(model, plot='pr', display_format='streamlit')

except Exception as e:
    st.error(f"Помилка при завантаженні моделі: {e}")
