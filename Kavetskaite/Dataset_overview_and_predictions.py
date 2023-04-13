import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data


def setup_inputs(*, input_parameters, columns: int = 4) -> dict:
    inputs = {}

    for i, column in enumerate(st.columns(columns)):
        with column:
            for j in range(i, len(input_parameters), columns):
                inputs[input_parameters[j]["label"]] = st.number_input(**input_parameters[j])
    return inputs


def predict(*, model, inputs: dict) -> float:
    wine = pd.DataFrame(inputs, index=[0])
    prediction = predict_model(model, data=wine)
    return prediction["Label"][0]


def display_wine_quality(*, predicted_wine_quality: float) -> None:
    if predicted_wine_quality < 5:
        color = "red"
    elif predicted_wine_quality < 8:
        color = "orange"
    else:
        color = "green"

    st.markdown(f"Predicted wine quality: :{color}[{predicted_wine_quality}]")


def add_categorical_true_false(*, inputs: dict, label: str, select: str) -> None:
    is_chosen = select == "Yes"
    inputs[label] = int(is_chosen)


def main():
    st.set_page_config(
        page_title="Predict Patient Survival",
        layout="wide",
    )

    st.header("Датасет")

    dataset = get_data("hepatitis")
    st.write(dataset)

    st.subheader("Опис ознак")
    
    columns_description = {
        "Class": "Patient survival (1 = died, 2 = survived)",
        "AGE": "Patient age",
        "SEX": "1 - male, 2 - female",
        "STEROID": "1 - no, 2 - yes",
        "ANTIVIRALS": "1 - no, 2 - yes",
        "FATIGUE": "1 - no, 2 - yes",
        "MALAISE": "1 - no, 2 - yes",
        "ANOREXIA": "1 - no, 2 - yes",
        "LIVER BIG": "1 - no, 2 - yes",
        "LIVER FIRM": "1 - no, 2 - yes",
        "SPLEEN PALPABLE": "1 - no, 2 - yes",
        "SPIDERS": "1 - no, 2 - yes",
        "ASCITES": "1 - no, 2 - yes",
        "VARICES": "1 - no, 2 - yes",
        "BILIRUBIN": "Bilirubin in mg/dL",
        "ALK PHOSPHATE": "Alkaline phosphatase in IU/L",
        "SGOT": "SGOT (serum glutamic-oxaloacetic transaminase) in U/L",
        "ALBUMIN": "Albumin in g/dL",
        "PROTIME": "Prothrombin time in seconds",
        "HISTOLOGY": "1 - no, 2 - yes",
    }

    n_of_cols = 4
    cols = st.columns(n_of_cols)
    for i, (column, description) in enumerate(columns_description.items()):
        with cols[i % n_of_cols]:
            with st.expander(column):
                st.write(description)

    st.subheader("Загальна інформація про датасет")
    st.write(dataset.describe())

    st.subheader("Передбачення")

    age = st.number_input("Age", min_value=0, max_value=100, value=0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    steroid = st.selectbox("Steriod", ["No", "Yes"])
    antivirals = st.selectbox("Antivirals", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    malaise = st.selectbox("Malaise", ["No", "Yes"])
    anorexia = st.selectbox("Anorexia", ["No", "Yes"])
    liver_big = st.selectbox("Liver big", ["No", "Yes"])
    liver_firm = st.selectbox("Liver firm", ["No", "Yes"])
    spleen_palpable = st.selectbox("Spleen palpable", ["No", "Yes"])
    spiders = st.selectbox("Spiders", ["No", "Yes"])
    ascites = st.selectbox("Ascites", ["No", "Yes"])
    varices = st.selectbox("Varices", ["No", "Yes"])
    bilirubin = st.number_input("Bilirubin", min_value=0.0, max_value=10.0, value=0.0)
    alk_phosphate = st.number_input("Alk phosphate", min_value=0, max_value=1000, value=0)
    sgot = st.number_input("SGOT", min_value=0, max_value=1000, value=0)
    albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=0.0)
    protime = st.number_input("Protime", min_value=0, max_value=1000, value=0)
    histology = st.selectbox("Histology", ["No", "Yes"])

    button_predict = st.button("Predict")

    if button_predict:
        inputs = {
            "age": age,
            "bilirubin": bilirubin,
            "alk_phosphate": alk_phosphate,
            "sgot": sgot,
            "albumin": albumin,
            "protime": protime,
        }
        add_categorical_true_false(inputs=inputs, label="sex", select=sex)
        add_categorical_true_false(inputs=inputs, label="steroid", select=steroid)
        add_categorical_true_false(inputs=inputs, label="antivirals", select=antivirals)
        add_categorical_true_false(inputs=inputs, label="fatigue", select=fatigue)
        add_categorical_true_false(inputs=inputs, label="malaise", select=malaise)
        add_categorical_true_false(inputs=inputs, label="anorexia", select=anorexia)
        add_categorical_true_false(inputs=inputs, label="liver_big", select=liver_big)
        add_categorical_true_false(inputs=inputs, label="liver_firm", select=liver_firm)
        add_categorical_true_false(inputs=inputs, label="spleen_palpable", select=spleen_palpable)
        add_categorical_true_false(inputs=inputs, label="spiders", select=spiders)
        add_categorical_true_false(inputs=inputs, label="ascites", select=ascites)
        add_categorical_true_false(inputs=inputs, label="varices", select=varices)
        add_categorical_true_false(inputs=inputs, label="histology", select=histology)

        model = load_model("hepatitis-classification-model")
        data = pd.DataFrame(inputs, index=[0])

        predicted_class = predict_model(model, data=data)

        survival = "виживе" if predicted_class['Label'][0] == 2 else "помере"
        probability = predicted_class['Score'][0] * 100

        st.write(f"Пацієнт {survival} з ймовірністю в {probability:.2f}%")


if __name__ == "__main__":
    main()
