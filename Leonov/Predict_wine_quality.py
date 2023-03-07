import streamlit as st

import pandas as pd

from pycaret.regression import (
    load_model,
    predict_model
)

from constants import (
    PAGE_CONFIG,
    INPUT_PARAMETERS
)

from typing import List


def setup_inputs(*, input_parameters: List[dict], columns: int = 4) -> dict:
    inputs = {}

    for i, column in enumerate(st.columns(columns)):
        with column:
            for j in range(i, len(input_parameters), columns):
                inputs[input_parameters[j]['label']] = st.number_input(**input_parameters[j])
    return inputs


def predict_wine_quality(*, model, inputs: dict) -> float:
    wine = pd.DataFrame(inputs, index=[0])
    prediction = predict_model(model, data=wine)
    return prediction['Label'][0]


def display_wine_quality(*, predicted_wine_quality: float) -> None:
    if predicted_wine_quality < 5:
        color = 'red'
    elif predicted_wine_quality < 8:
        color = 'orange'
    else:
        color = 'green'

    st.markdown(f'Predicted wine quality: :{color}[{predicted_wine_quality}]')


@st.cache_data
def cached_load_model(*, model_name: str):
    model = load_model(model_name)
    return model


def main():
    # setup page
    st.set_page_config(
        page_title='Predict Wine Quality',
        **PAGE_CONFIG
    )

    # get model
    model = cached_load_model(model_name='wine-quality-prediction-pipeline')

    # setup main page
    st.header('Wine Quality Prediction')
    st.subheader('Input wine parameters to predict its quality')

    inputs = setup_inputs(input_parameters=INPUT_PARAMETERS)
    button_predict = st.button('Predict')

    if button_predict:
        predicted_wine_quality = predict_wine_quality(
            model=model,
            inputs=inputs
        )
        display_wine_quality(predicted_wine_quality=predicted_wine_quality)


if __name__ == '__main__':
    main()
