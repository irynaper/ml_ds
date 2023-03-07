import streamlit as st
from pycaret.datasets import get_data


@st.cache_data
def cached_get_data(*, dataset_name: str):
    dataset = get_data(dataset_name)
    return dataset


def setup_sidebar(*, title: str):
    sidebar = st.sidebar
    sidebar.title(title)
    return sidebar
