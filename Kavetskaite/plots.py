import altair as alt


def alt_bar_chart_count(
        *,
        dataset, x: str,
        type: str = 'quantitative',
        title: str = None,
        binned: bool = False,
        maxbins: int = 100
):
    chart = (
        alt
        .Chart(dataset)
        .encode(
            x=alt.X(
                x,
                type=type,
                axis=alt.Axis(
                    title=x.capitalize(),
                    labelAngle=0
                ),
                bin=alt.Bin(binned=binned, maxbins=maxbins)
            ),
            y=alt.Y(
                'count(*):Q',
                axis=alt.Axis(title='Count')
            ),
        )
        .mark_bar()
    )
    if title:
        chart = chart.properties(title=title)
    return chart


def create_correlation_dataframe(*, dataset):
    cor = (
        dataset
        .corr()
        .stack()
        .reset_index()
        .rename(
            columns={
                'level_0': 'feature_1',
                'level_1': 'feature_2',
                0: 'correlation'
            }
        )
    )
    return cor


def create_heatmap(*, correlation_dataframe):
    # base chart
    base = (
        alt
        .Chart(correlation_dataframe)
        .encode(
            y=alt.Y(
                field='feature_1',
                title=''
            ),
            x=alt.X(
                field='feature_2',
                title=''
            )
        )
        .properties(
            width=600,
            height=525
        )
    )

    # correlation heatmap
    heatmap = (
        base
        .mark_rect()
        .encode(
            color=alt.Color(
                'correlation:Q',
                scale=alt.Scale(scheme='reds'),
            )
        )
    )

    # labels
    text = (
        base
        .mark_text()
        .encode(
            text=alt.Text(
                'correlation:Q',
                format='.2f'
            ),
            color=alt.value('white')
        )
    )

    return heatmap, text


def correlation_with_quality_plot(*, dataset, feature: str):
    base = (
        alt
        .Chart(
            data=dataset,
            title=f'Correlation between {feature.capitalize()} and Quality'
        )
        .encode(
            x=alt.X(
                'quality:N',
                axis=alt.Axis(
                    title='Quality',
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                feature,
                axis=alt.Axis(title=feature.capitalize()),
                scale=alt.Scale(zero=False)
            ),
        )
        .mark_point()
    )

    chart = (
        base
        .transform_regression('quality', feature)
        .mark_line()
        .encode(
            color=alt.value('green')
        )
    )

    return base + chart