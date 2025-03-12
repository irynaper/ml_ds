import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pycaret.regression import load_model, predict_model
from pycaret.datasets import get_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

clarity_order = ['SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']
cut_order = ['Fair', 'Good', 'Very Good', 'Ideal', 'Signature-Ideal']
color_order = ['I', 'H', 'G', 'F', 'E', 'D']
polish_order = ['G', 'VG', 'EX', 'ID']
symmetry_order = ['G', 'VG', 'EX', 'ID']

tab1, tab2, tab3 = st.tabs(["Опис датасету", "Візуалізації", "Прогнозування"])

@st.cache_data
def load_data():
    return get_data('diamond')

# |-------------|
# |Опис датасету|
# |-------------|
def display_data_description(data):
    st.image('https://u.osu.edu/diamondscarlsoncaggiano/files/2015/04/diamonds-background-271garm.jpg', use_container_width=True,)    

    st.title('Опис датасету "Diamond"')
    st.subheader('ММН, ПЗ №2, КНТу-22-1, Сургай А.')
    st.subheader('Опис ознак')
    st.markdown("""
    1) **Carat Weight (вага в каратах)** – вага діаманта, вимірюється в каратах (1 карат = 0,2 г).
    2) **Cut (огранювання)** – якість огранювання, яка визначає, наскільки добре діамант відбиває світло.
    Основні категорії:
        - Fair (Посереднє)
        - Good (Добре)
        - Very Good (Дуже добре)
        - Ideal (Ідеальне)
        - Signature-Ideal (Фірмово-ідеальний)
    3) **Color (колір)** – оцінка кольору діаманта за шкалою від I (помітний жовтуватий відтінок - *найгірший*) до D (абсолютно безбарвний - *найкращий*). Безбарвні діаманти вважаються найціннішими. (I, H, G, F, E, D)
    4) **Clarity (чистота)** – відображає наявність внутрішніх і зовнішніх дефектів.
    Основні категорії:
        - SI1 (Slightly Included 1) – незначні включення, помітні при 10-кратному збільшенні.
        - VS2 (Very Slightly Included 2) – дуже незначні включення, які трохи помітні під 10-кратним збільшенням.
        - VS1 (Very Slightly Included 1) – дуже незначні включення, менш помітні, ніж у VS2.
        - VVS2 (Very Very Slightly Included 2) – дуже-дуже незначні включення, важко помітні навіть під 10-кратним збільшенням.
        - VVS1 (Very Very Slightly Included 1) – ще чистіший рівень, майже непомітні включення.
        - IF (Internally Flawless) – немає внутрішніх дефектів, лише незначні зовнішні дефекти.
        - FL (Flawless) – ідеальна чистота, без жодних включень або дефектів навіть при 10-кратному збільшенні.
    5) **Polish (полірування)** – якість полірування граней діаманта. Чим краще полірування, тим більше світла проходить через діамант.
    Оцінюється як:
        - G (Good) (Добре)
        - VG (Very Good) (Дуже добре)
        - EX (Excellent) (Чудове)
        - ID (Ideal) (Ідеальне)
    6) **Symmetry (симетрія)** – наскільки рівномірно вирізані грані діаманта. Симетрія впливає на розсіювання світла та блиск.
    Оцінюється як:
        - G (Good) (Добре)
        - VG (Very Good) (Дуже добре)
        - EX (Excellent) (Чудове)
        - ID (Ideal) (Ідеальне)
    7) **Report (сертифікат/звіт)** – сертифікація діаманта від незалежних лабораторій, таких як GIA (Gemological Institute of America), AGSL (American Gem Society Laboratories).
    8) **Price (ціна)** – вартість діаманта в доларах США.
    """)
    
    st.subheader("Дані датасету")
    with st.expander("Дані датасету", expanded=True):
        st.dataframe(data)
        
    st.subheader("Статистичний аналіз числових характеристик")
    with st.expander("Статистичний аналіз числових характеристик", expanded=True):
        st.write(data.describe())

# |-------------|
# | Візуалізації|
# |-------------|
def plot_feature_distribution(data):
    st.title("Візуалізації")
    st.subheader("Розподіл діамантів за обраною ознакою")
    selected_feature = st.selectbox("Оберіть ознаку :", ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report', 'Price'])

    if selected_feature in ['Price', 'Carat Weight']:
        if selected_feature == 'Price':
            bins = [0, 1000, 5000, 10000, 20000, 50000, 100000]
            labels = ["<1000", "1000-5000", "5000-10000", "10000-20000", "20000-50000", ">50000"]
            data['Price Range'] = pd.cut(data['Price'], bins=bins, labels=labels)
            feature_dist = data['Price Range'].value_counts()
            feature_dist = feature_dist.reindex(labels)
        else:
            bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
            labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "4-5"]
            data['Carat Weight Range'] = pd.cut(data['Carat Weight'], bins=bins, labels=labels)
            feature_dist = data['Carat Weight Range'].value_counts()
            feature_dist = feature_dist.reindex(labels)
    else:
        feature_dist = data[selected_feature].value_counts()
        if selected_feature == 'Cut':
            feature_dist = feature_dist.reindex(cut_order)
        elif selected_feature == 'Color':
            feature_dist = feature_dist.reindex(color_order)
        elif selected_feature == 'Clarity':
            feature_dist = feature_dist.reindex(clarity_order)
        elif selected_feature == 'Polish':
            feature_dist = feature_dist.reindex(polish_order)
        elif selected_feature == 'Symmetry':
            feature_dist = feature_dist.reindex(symmetry_order)

    fig = px.bar(feature_dist, 
                 x=feature_dist.index, 
                 y=feature_dist.values, 
                 title=f'Кількість діамантів по {selected_feature}', 
                 labels={selected_feature: f"{selected_feature}", "value": "Кількість діамантів"})
    fig.update_layout(yaxis_title="Count")
    fig.update_traces(hovertemplate='%{x}: %{y} діамантів')
    st.plotly_chart(fig)
    
def plot_carat_weight_and_price_distribution(data):
    st.subheader("Розподіл діамантів по діапазонам числових ознак")
    col1, col2 = st.columns(2)
    
    bins_carat = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
    labels_carat = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "4-5"]
    
    data['Carat Weight Range'] = pd.cut(data['Carat Weight'], bins=bins_carat, labels=labels_carat)
    
    carat_weight_distribution = data['Carat Weight Range'].value_counts()

    with col1:
        fig_carat = px.pie(values=carat_weight_distribution.values, 
                           names=carat_weight_distribution.index, 
                           title="Розподіл діамантів по діапазонах Carat Weight",
                           labels={"values": "Кількість діамантів", "names": "Діапазон Carat Weight"})
        st.plotly_chart(fig_carat)

    bins_price = [0, 1000, 5000, 10000, 20000, 50000, 100000]
    labels_price = ["<1000", "1000-5000", "5000-10000", "10000-20000", "20000-50000", ">50000"]
    
    data['Price Range'] = pd.cut(data['Price'], bins=bins_price, labels=labels_price)
    
    price_distribution = data['Price Range'].value_counts()

    with col2:
        fig_price = px.pie(values=price_distribution.values, 
                           names=price_distribution.index, 
                           title="Розподіл діамантів по діапазонах Price",
                           labels={"values": "Кількість діамантів", "names": "Діапазон Price"})
        st.plotly_chart(fig_price)

def plot_characteristics_distribution(data):
    st.subheader("Розподіл діамантів за категоріальними ознаками")
    
    characteristics = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
    
    col1, col2 = st.columns(2)

    for i, characteristic in enumerate(characteristics):
        dist = data[characteristic].value_counts()

        if i % 2 == 0:
            with col1:
                fig = px.pie(values=dist.values, 
                             names=dist.index, 
                             title=f"Розподіл діамантів за значенням {characteristic}",
                             labels={"values": "Кількість діамантів", "names": characteristic})
                st.plotly_chart(fig)
        else:
            with col2:
                fig = px.pie(values=dist.values, 
                             names=dist.index, 
                             title=f"Розподіл діамантів за значенням {characteristic}",
                             labels={"values": "Кількість діамантів", "names": characteristic})
                st.plotly_chart(fig)

def plot_price_vs_carat(data):
    st.subheader("Залежність Price від Carat Weight")

    min_carat, max_carat = st.slider(
        "Оберіть діапазон Carat Weight:",
        min_value=float(data['Carat Weight'].min()),
        max_value=float(data['Carat Weight'].max()),
        value=(float(data['Carat Weight'].min()), float(data['Carat Weight'].max()))
    )

    min_price, max_price = st.slider(
        "Оберіть діапазон Price:",
        min_value=float(data['Price'].min()),
        max_value=float(data['Price'].max()),
        value=(float(data['Price'].min()), float(data['Price'].max()))
    )

    filtered_data = data[(data['Carat Weight'] >= min_carat) & (data['Carat Weight'] <= max_carat) &
                         (data['Price'] >= min_price) & (data['Price'] <= max_price)]

    fig = px.scatter(filtered_data, x='Carat Weight', y='Price', trendline='ols', title="Залежність Price від Carat Weight")
    st.plotly_chart(fig)

def plot_price_by_feature(data):
    st.subheader("Розподіл Price залежно від обраної ознаки")
    
    feature = st.selectbox("Оберіть характеристику:", 
                           ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry'], 
                           key="feature_select")
    
    if feature == 'Cut':
        order = cut_order
    elif feature == 'Color':
        order = color_order
    elif feature == 'Clarity':
        order = clarity_order
    elif feature == 'Polish':
        order = polish_order
    elif feature == 'Symmetry':
        order = symmetry_order
    else:
        order = None
    
    levels = st.multiselect(f"Оберіть рівні {feature}:", 
                            data[feature].unique(), 
                            default=data[feature].unique(), 
                            key=f"levels_{feature}")
    
    filtered_data = data[data[feature].isin(levels)]
    
    fig = px.box(filtered_data, 
                 x=feature, 
                 y='Price', 
                 title=f'Розподіл Price залежно від {feature}', 
                 category_orders={feature: order} if order else None)
    
    st.plotly_chart(fig)
    
def plot_correlation_heatmap(data):
    st.subheader("Теплова карта кореляцій числових змінних")
    
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    corr = data[numerical_features].corr()
    
    fig = px.imshow(corr, 
                    labels=dict(x="Ознаки", y="Ознаки", color="Кореляція"),
                    x=corr.columns,
                    y=corr.columns,
                    title="Теплова карта кореляцій")
    st.plotly_chart(fig)
    
def plot_violin_price_by_feature(data):
    st.subheader("Розподіл ціни залежно від обраної ознаки")
    
    feature = st.selectbox("Оберіть ознаку:", ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry'])
    if feature == 'Cut':
        order = cut_order
    elif feature == 'Color':
        order = color_order
    elif feature == 'Clarity':
        order = clarity_order
    elif feature == 'Polish':
        order = polish_order
    elif feature == 'Symmetry':
        order = symmetry_order
    
    fig = px.violin(data, 
                    x=feature, 
                    y='Price', 
                    box=True, 
                    points="all", 
                    title=f"Розподіл ціни залежно від {feature}",
                    category_orders={feature: order})
    
    st.plotly_chart(fig)

def plot_3d_scatter(data):
    st.subheader("3D діаграма розсіювання для порівняння Carat Weight, Price та інших ознак")
    
    feature_z = st.selectbox("Оберіть характеристику для осі Z:", 
                             ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report'], 
                             key="feature_z")
    
    feature_color = st.selectbox("Оберіть характеристику для кольорового кодування:", 
                                 ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report'], 
                                 key="feature_color")
    
    fig = px.scatter_3d(data, 
                        x='Carat Weight', 
                        y='Price', 
                        z=feature_z, 
                        color=feature_color, 
                        title=f"3D діаграма розсіювання: Carat Weight, Price, {feature_z} (колір: {feature_color})",
                        labels={'Carat Weight': 'Вага (карат)', 'Price': 'Ціна', feature_z: feature_z, feature_color: feature_color})
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Carat Weight',
            yaxis_title='Price',
            zaxis_title=feature_z
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig)
    
def plot_pair_plot(data):
    st.subheader("Матриця діаграм розсіювання")
    
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    fig = px.scatter_matrix(data[numerical_features], 
                            title="Матриця діаграм розсіювання для числових змінних")
    st.plotly_chart(fig)
    
def plot_facet_grid_universal(data):
    st.subheader("Фасетні графіки: Price vs Carat Weight залежно від обраної ознаки")
    
    feature = st.selectbox("Оберіть ознаку для фасетних графіків:", 
                           ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report'])
    
    if feature == 'Cut':
        order = cut_order
    elif feature == 'Color':
        order = color_order
    elif feature == 'Clarity':
        order = clarity_order
    elif feature == 'Polish':
        order = polish_order
    elif feature == 'Symmetry':
        order = symmetry_order
    else:
        order = None
    
    fig = px.scatter(data, 
                     x='Carat Weight', 
                     y='Price', 
                     color=feature, 
                     facet_col=feature, 
                     title=f"Price vs Carat Weight залежно від {feature}",
                     category_orders={feature: order} if order else None)
    
    st.plotly_chart(fig, key=f"facet_grid_{feature}")
    
def plot_hexbin(data):
    st.subheader("Щільність розподілу Carat Weight та Price")
    
    fig = px.density_heatmap(data, 
                             x='Carat Weight', 
                             y='Price', 
                             nbinsx=20, 
                             nbinsy=20, 
                             title="Щільність розподілу Carat Weight та Price")
    st.plotly_chart(fig)
    
def plot_radar_chart(data):
    st.subheader("Радарний графік для порівняння характеристик діамантів")
    
    diamond_index_1 = st.selectbox("Оберіть індекс першого діаманта:", data.index, key="diamond_1")
    diamond_1 = data.loc[diamond_index_1]
    
    diamond_index_2 = st.selectbox("Оберіть індекс другого діаманта:", data.index, key="diamond_2")
    diamond_2 = data.loc[diamond_index_2]
    
    categories = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry']
    values_1 = [diamond_1[cat] for cat in categories]
    values_2 = [diamond_2[cat] for cat in categories]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_1,
        theta=categories,
        fill='toself',
        name=f"Діамант {diamond_index_1}"
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_2,
        theta=categories,
        fill='toself',
        name=f"Діамант {diamond_index_2}"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, len(categories)]
            )),
        showlegend=True,
        title="Порівняння характеристик двох діамантів"
    )

    st.plotly_chart(fig)

# |-------------|
# |Прогнозування|
# |-------------|
def plot_actual_vs_predicted_price_comparison(predictions):
    st.subheader('Порівняння фактичних та прогнозованих цін')
    
    # Додавання кнопок для перемикання між типами графіків
    plot_type = st.radio("Оберіть тип графіка:", ('Стовпчикова діаграма', 'Лінійний графік'))

    fig = go.Figure()

    if plot_type == 'Стовпчикова діаграма':
        fig.add_trace(go.Bar(
            x=predictions.index,
            y=predictions['Price'],
            name='Фактична ціна',
            marker_color='skyblue',
            hovertemplate="Індекс: %{x}<br>Ціна: %{y}<extra></extra>"
        ))

        fig.add_trace(go.Bar(
            x=predictions.index,
            y=predictions['prediction_label'],
            name='Прогнозована ціна',
            marker_color='salmon',
            hovertemplate="Індекс: %{x}<br>Ціна: %{y}<extra></extra>"
        ))

        fig.update_layout(
            title="Порівняння фактичних та прогнозованих цін",
            xaxis_title="Індекс діаманта",
            yaxis_title="Ціна",
            barmode='group',
            hovermode="x unified",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
    else:
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions['Price'],
            name='Фактична ціна',
            mode='lines+markers',
            line=dict(color='skyblue'),
            hovertemplate="Індекс: %{x}<br>Ціна: %{y}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions['prediction_label'],
            name='Прогнозована ціна',
            mode='lines+markers',
            line=dict(color='salmon'),
            hovertemplate="Індекс: %{x}<br>Ціна: %{y}<extra></extra>"
        ))

        fig.update_layout(
            title="Порівняння фактичних та прогнозованих цін",
            xaxis_title="Індекс діаманта",
            yaxis_title="Ціна",
            hovermode="x unified",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

    st.plotly_chart(fig)

def plot_actual_vs_predicted_scatter(predictions):
    fig = px.scatter(predictions, 
                     x='Price', 
                     y='prediction_label', 
                     title="Фактична ціна vs Прогнозована ціна",
                     labels={'Price': 'Фактична ціна', 'prediction_label': 'Прогнозована ціна'},
                     trendline='ols')
    fig.update_traces(marker=dict(size=10, opacity=0.6))
    st.plotly_chart(fig)
    
def plot_residuals(predictions):
    st.subheader('Графік залишків')
    
    predictions['Residuals'] = predictions['Price'] - predictions['prediction_label']
    fig = px.scatter(predictions, 
                     x='prediction_label', 
                     y='Residuals', 
                     title="Графік залишків",
                     labels={'prediction_label': 'Прогнозована ціна', 'Residuals': 'Залишки'},
                     hover_data={'prediction_label': ':.2f', 'Residuals': ':.2f', 'Price': ':.2f'})  # Додаткові дані для підказок

    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig)
    
def plot_cumulative_error(predictions):
    st.subheader('Кумулятивна похибка')
    
    predictions['Cumulative Error'] = (predictions['Price'] - predictions['prediction_label']).cumsum()
    fig = px.line(predictions, 
                  x=predictions.index, 
                  y='Cumulative Error', 
                  title="Кумулятивна похибка",
                  labels={'index': 'Індекс діаманта', 'Cumulative Error': 'Кумулятивна похибка'})
    st.plotly_chart(fig)
    
def plot_feature_importance(model):
    st.subheader('Важливість ознак для прогнозування ціни')
    
    importances = model.feature_importances_
    features = ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Додавання випадаючого меню для вибору ознак
    selected_features = st.multiselect("Оберіть ознаки для відображення:", features, default=features)

    filtered_importance_df = importance_df[importance_df['Feature'].isin(selected_features)]

    fig = px.bar(filtered_importance_df, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h', 
                 title="Важливість ознак для прогнозування ціни",
                 labels={'Importance': 'Важливість', 'Feature': 'Ознака'})
    st.plotly_chart(fig)

def plot_error_distribution(predictions):
    st.subheader('Розподіл похибок')
    
    errors = predictions['Price'] - predictions['prediction_label']
    fig = px.histogram(errors, 
                       nbins=30, 
                       title="Розподіл похибок",
                       labels={'value': 'Похибка', 'count': 'Кількість'})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

def evaluate_model(predictions):
    st.subheader('Метрики')
    mae = mean_absolute_error(predictions['Price'], predictions['prediction_label'])
    rmse = mean_squared_error(predictions['Price'], predictions['prediction_label'], squared=False)
    r2 = r2_score(predictions['Price'], predictions['prediction_label'])
    mape = mean_absolute_percentage_error(predictions['Price'], predictions['prediction_label'])
    
    st.write(f"**MAE (Середня абсолютна похибка):** {mae:.2f}")
    st.write(f"**RMSE (Середньоквадратична похибка):** {rmse:.2f}")
    st.write(f"**R² (Коефіцієнт детермінації):** {r2:.2f}")
    st.write(f"**MAPE (Середня абсолютна процентна похибка):** {mape:.2%}")

def load_and_predict_model(model_path, data):
    st.title("Прогнозування ціни діаманта")

    st.subheader("Завантаження моделі")
    try:
        model = load_model(model_path)
        st.success("Модель успішно завантажена!")
    except Exception as e:
        st.error(f"Не вдалося завантажити модель: {e}")
        model = None

    if model:
        st.subheader("Перегляд датасету з прогнозованою ціною")
        sample_size = st.slider("Оберіть кількість зразків для пронозування:", 10, 100, 50)
        sample_data = data.sample(sample_size)
        predictions = predict_model(model, data=sample_data)
        predictions.reset_index(drop=True, inplace=True)
        
        with st.expander("Набір даних з прогнозами", expanded=True):
            st.write(predictions[['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report', 'Price', 'prediction_label']])

        plot_actual_vs_predicted_price_comparison(predictions)
        plot_actual_vs_predicted_scatter(predictions)
        plot_residuals(predictions)
        plot_cumulative_error(predictions)
        plot_feature_importance(model)
        plot_error_distribution(predictions)
        evaluate_model(predictions)

        st.subheader("Прогноз ціни для нового діаманта")
        carat = st.slider("Вага в каратах:", 0.2, 5.0, 1.0)
        col1, col2 = st.columns(2)
        with col1:
            cut = st.selectbox("Тип огранювання:", cut_order)
            color = st.selectbox("Колір:", color_order)
            clarity = st.selectbox("Чистота:", clarity_order)
        
        with col2:
            polish = st.selectbox("Полірування:", polish_order)
            symmetry = st.selectbox("Симетрія:", symmetry_order)
            report = st.selectbox("Звіт:", data['Report'].unique()) 

        new_data = pd.DataFrame({
            'Carat Weight': [carat],
            'Cut': [cut],
            'Color': [color],
            'Clarity': [clarity],
            'Polish': [polish],
            'Symmetry': [symmetry],
            'Report': [report],
        })

        prediction = predict_model(model, data=new_data)
        st.write(f"Прогнозована ціна: {prediction['prediction_label'][0]:.2f} $")

# |-------------|
# |     Main    |
# |-------------|
def main():
    data = load_data()

    with tab1:
        display_data_description(data)

    with tab2:
         plot_feature_distribution(data)
         plot_carat_weight_and_price_distribution(data)
         plot_characteristics_distribution(data)
         plot_price_vs_carat(data)
         plot_price_by_feature(data)
         plot_correlation_heatmap(data)
         plot_violin_price_by_feature(data)
         plot_3d_scatter(data)
         plot_pair_plot(data)
         plot_facet_grid_universal(data)
         plot_hexbin(data)
         plot_radar_chart(data)
         
    with tab3:
        model_path = "diamond-pipeline"
        load_and_predict_model(model_path, data)

if __name__ == '__main__':
    main()