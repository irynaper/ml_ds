import streamlit as st
from pycaret.datasets import get_data
import altair as alt


def alt_bar(*, dataset, x: str, type: str):
    chart = (
        alt
        .Chart(dataset)
        .mark_bar()
        .encode(
            x=alt.X(x, type=type, axis=alt.Axis(labelAngle=0)),
            y="count()",
        )
    )
    return chart


def main():
    st.set_page_config(
        page_title="Visualisations",
        layout="centered"
    )
    st.title("Візуалізації даних та короткий опис")
    st.subheader("Розподіл ознак")

    dataset = get_data("hepatitis")

    st.write("Можемо побачити, що у датасеті переважають випадки, де пацієнт помер")
    chart = alt_bar(dataset=dataset, x="Class", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("Вік є нормально розподіленим")
    chart = alt_bar(dataset=dataset, x="AGE", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("Стероїди використовуються в половині випадків ")
    chart = alt_bar(dataset=dataset, x="STEROID", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості випадків використовуються антивірусні препарати")
    chart = alt_bar(dataset=dataset, x="ANTIVIRALS", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    chart = alt_bar(dataset=dataset, x="FATIGUE", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    chart = alt_bar(dataset=dataset, x="MALAISE", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("Більшість пацієнтів мають анорексію")
    chart = alt_bar(dataset=dataset, x="ANOREXIA", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("Більшість пацієнтів мають велику печінку, при цьому є пропущені значення")
    chart = alt_bar(dataset=dataset, x="LIVER BIG", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("Більшість пацієнтів мають 'тверду' печінку, при цьому є пропущені значення")
    chart = alt_bar(dataset=dataset, x="LIVER FIRM", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів селезінка пальпується. За звичай це не так, за винятком худорлявих молодих людей")
    chart = alt_bar(dataset=dataset, x="SPLEEN PALPABLE", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів є павутинні ангіоми (павукоподібні кровоносні судини на шкірі), при цьому є пропущені значення")
    chart = alt_bar(dataset=dataset, x="SPIDERS", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів є асцит, при цьому є пропущені значення")
    chart = alt_bar(dataset=dataset, x="ASCITES", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів є варикоз, при цьому є пропущені значення")
    chart = alt_bar(dataset=dataset, x="VARICES", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів білірубін знаходиться у межах норми, яка є від 0.1 до 1.2 mg/dL")
    chart = alt_bar(dataset=dataset, x="BILIRUBIN", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів алкфосфатаза знаходиться у межах норми, яка є від 44 до 147 IU/L")
    chart = alt_bar(dataset=dataset, x="ALK PHOSPHATE", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("У більшості пацієнтів рівень SGOT знаходиться поза нормою, яка є від 8 до 45 U/L")
    chart = alt_bar(dataset=dataset, x="SGOT", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("У багатьох пацієнтів занижений альбумін, норма якого є від 3.5 до 5.5 g/dL")
    chart = alt_bar(dataset=dataset, x="ALBUMIN", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("У всіх пацієнтів протромбіновий час знаходиться поза нопмою, яка є від 11 до 13.5 s")
    chart = alt_bar(dataset=dataset, x="PROTIME", type="quantitative")
    st.altair_chart(chart, use_container_width=True)

    st.write("У половини пацієнтів наявна гістологія")
    chart = alt_bar(dataset=dataset, x="HISTOLOGY", type="nominal")
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Теплова карта")
    cor = dataset.corr().stack().reset_index()

    base_chart = (
        alt
        .Chart(cor)
        .encode(
            y=alt.Y(field="level_0", title="Feature 1"),
            x=alt.X(field="level_1", title="Feature 2")
        )
        .properties(width=900, height=900)
    )

    heatmap = (
        base_chart
        .mark_rect()
        .encode(
            color=alt.Color("0:Q", scale=alt.Scale(scheme="purples"), title="Correlation")
        )
    )

    text = (
        base_chart
        .mark_text()
        .encode(
            text=alt.Text("0:Q", format=".2f"),
            color=alt.value("black")
        )
    )

    st.write("З теплової карти можна виділити кореляції між ознаками.")
    st.write("Окрім очевидних, таких як загальне почуття дискомфорту (MALAISE) та сильною втомою (FATIGUE), що мають кореляцію 0.6. "
             "Бачимо, що багато ознак досить сильно корелюють між собою як позитивно, так і негативно.")
    st.altair_chart(heatmap + text)


if __name__ == "__main__":
    main()
