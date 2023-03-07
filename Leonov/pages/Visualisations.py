import streamlit as st

from plots import *

from utils import (
    cached_get_data,
    setup_sidebar
)

from constants import PAGE_CONFIG


def main():
    # setup page
    st.set_page_config(
        page_title='Visualisations',
        page_icon=PAGE_CONFIG['page_icon'],
        layout='centered'
    )
    st.title('Visualisations')

    # setup sidebar
    sidebar = setup_sidebar(title='Contents')

    # setup sidebar links
    links = [
        'Distributions',
        'Distribution Analysis',
        'Correlation Heatmap',
        'Positive Correlations',
        'Negative Correlations'
    ]
    for link in links:
        sidebar.markdown(f'[{link}](#{link.lower().replace(" ", "-")})')

    # get dataset
    dataset = cached_get_data(dataset_name='wine')

    # bar charts
    st.header(links[0])

    # red / white wine
    chart = alt_bar_chart_count(
        dataset=dataset,
        x='type',
        type='nominal',
        title='Wine type distribution',
        binned=True,
        maxbins=2
    )
    st.altair_chart(chart, use_container_width=True)

    # wine quality distribution
    chart = alt_bar_chart_count(
        dataset=dataset,
        x='quality',
        type='nominal',
        title='Wine quality distribution',
        binned=True,
    )
    st.altair_chart(chart, use_container_width=True)

    # other features distribution
    new_ds = dataset.drop(['quality', 'type'], axis=1)
    for feature in new_ds.columns:
        chart = alt_bar_chart_count(dataset=new_ds, x=feature, title=feature.capitalize() + ' distribution')
        st.altair_chart(chart, use_container_width=True)

    # analysis
    st.header(links[1])
    st.markdown(
        """
            <p style="font-size: 20px">
                We can see that all features have many outliers with only density and residual sugar 
                having only a few of them.
            </p>
        """,
        unsafe_allow_html=True
    )

    # correlation heatmap
    st.header(links[2])

    # create correlation dataframe
    cor = create_correlation_dataframe(dataset=dataset)

    # create heatmap
    heatmap, text = create_heatmap(correlation_dataframe=cor)
    st.altair_chart(heatmap + text)

    # apparent correlations with quality
    st.title('Correlation with Quality')

    features_with_positive_correlation = ['alcohol']
    features_with_negative_correlation = ['volatile acidity', 'chlorides', 'density']

    # positive correlations
    st.header(links[3])
    for feature in features_with_positive_correlation:
        chart = correlation_with_quality_plot(dataset=dataset, feature=feature)
        st.altair_chart(chart, use_container_width=True)

    # negative correlations
    st.header(links[4])
    for feature in features_with_negative_correlation:
        chart = correlation_with_quality_plot(dataset=dataset, feature=feature)
        st.altair_chart(chart, use_container_width=True)


if __name__ == '__main__':
    main()
