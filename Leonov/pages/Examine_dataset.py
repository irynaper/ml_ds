import streamlit as st

from utils import (
    cached_get_data,
    setup_sidebar
)

from constants import DESCRIPTIONS

from typing import (
    List,
    Dict
)


def describe_features(*, descriptions: List[Dict[str, str]]):
    feature_descriptions = [
        {
            'label': feature['label'],
            'description': feature['description']
        }
        for feature in descriptions
    ]
    tabs = st.tabs([feature['label'] for feature in feature_descriptions])

    for i, tab in enumerate(tabs):
        with tab:
            st.write(feature_descriptions[i]['description'])


def main():
    # setup page
    st.set_page_config(
        page_title='Examine Dataset',
        page_icon='🍷',
        layout='wide',
    )

    # setup sidebar
    sidebar = setup_sidebar(title='Contents')

    # setup sidebar links
    links = [
        'Examine Dataset',
        'Features Description',
        'Dataset Statistics'
    ]
    for link in links:
        sidebar.markdown(f'[{link}](#{link.lower().replace(" ", "-")})')

    st.header(links[0])

    # get dataset
    dataset = cached_get_data(dataset_name='wine')
    st.write(dataset)

    # display feature descriptions
    st.header(links[1])
    describe_features(descriptions=DESCRIPTIONS)

    # display dataset statistics
    st.header(links[2])
    st.write(dataset.describe())


if __name__ == "__main__":
    main()
