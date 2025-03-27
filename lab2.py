# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 08:03:09 2025

@author: annanikolaichuk
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model, predict_model
from pycaret.datasets import get_data
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

sns.set_theme()

# Завантаження датасету в кеш, корекція помилок в колонках
@st.cache_data
def load_data():
    df = get_data('automobile')
    fix_col = ['bore', 'stroke', 'horsepower', 'peak-rpm']
    df[fix_col] = df[fix_col].apply(pd.to_numeric, errors='coerce')
    return df

# Завантаження моделі в кеш
@st.cache_resource
def load_trained_model():
    return load_model('automobile-pipeline')

def display_data_description():
    st.write("""
    The **Automobile dataset** is a comprehensive collection of car specifications,
    designed for predictive modeling and analysis of vehicle prices.
    It includes various technical and categorical attributes that influence a car's value,
    such as:
    - **symboling**: Symbolic number representing the risk factor of the car.
    - **normalized-losses**: Normalized loss value, used to compare the relative loss of the car.
    - **make**: The make of the automobile.
    - **fuel-type**: Type of fuel used by the automobile.
    - **aspiration**: Aspiration type of the engine (std/turbo).
    - **num-of-doors**: Number of doors of the automobile.
    - **body-style**: Body style of the automobile (convertible, sedan, etc.).
    - **drive-wheels**: Type of drive wheels (fwd, rwd, 4wd).
    - **engine-location**: Location of the engine (front or rear).
    - **wheel-base**: The distance between the front and rear axles.
    - **length**: Length of the automobile.
    - **width**: Width of the automobile.
    - **height**: Height of the automobile.
    - **curb-weight**: Weight of the automobile without passengers or cargo.
    - **engine-type**: Type of engine (dohc, ohcv, etc.).
    - **num-of-cylinders**: Number of cylinders in the engine.
    - **engine-size**: Size of the engine in cubic inches.
    - **fuel-system**: Type of fuel system (mpfi, 2bbl, etc.).
    - **bore**: Bore of the engine.
    - **stroke**: Stroke of the engine.
    - **compression-ratio**: Ratio of the cylinder's volume at the bottom of the piston stroke to the volume at the top.
    - **horsepower**: Engine horsepower.
    - **peak-rpm**: Maximum revolutions per minute of the engine.
    - **city-mpg**: Fuel efficiency in city driving (miles per gallon).
    - **highway-mpg**: Fuel efficiency in highway driving (miles per gallon).
    - **price**: Price of the automobile.
    """)

model = load_trained_model()
df = load_data()

# Розділення різних функцій по вкладкам: 
    # 1. Для візуалізацій початкового датасету
    # 2. Для візуалізації результатів навчання моделі з lab1
    # 3. Інтерактивна сторінка з вибором параметрів та відповідним прогнозом значення з моделі
    # 4. Загальна інформація про те, що можна робити у кожній вкладці
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Visualizations", "🤖 Model Predictions", "🎯 Interactive Prediction", "📖 How to Use"])

# Враховуючи розділення різної логіки по вкладкам, бокова панель не була створена,
# інтерактивні елементи різного призначення знаходяться в кожній вкладці окремо

with tab1:
    st.write("### 📊 Explore & Visualize the Automobile Dataset")
    st.markdown("""
    Get a deeper understanding of the automobile dataset through interactive visualizations.
    Use the tools below to explore numerical and categorical features, identify trends, and discover correlations.
    """)

    # Опис датасету
    display_data_description()

    st.markdown("""
    #### 🔍 Choose Your Visualization
    Select a **chart type** and relevant attributes to generate insightful graphs.
    """)
    
    # Вибір типу візуалізації
    viz_type = st.radio("Visualization Type", [
        "Histogram", "Boxplot", "Bar Plot", "Pie Chart", "Correlation Matrix"])

    # Вибір колонок для візуалізації, залежно від обраного типу (числові/категорії окремо)
    if viz_type in ["Histogram", "Boxplot"]:
        cols_to_display = st.multiselect("Select numerical columns", [
            'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height',
            'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
        ])
    elif viz_type in ["Bar Plot", "Pie Chart"]:
        cols_to_display = st.multiselect("Select categorical columns", [
            'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
            'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system'
        ])
    elif viz_type == "Correlation Matrix":
        cols_to_display = [
            'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height',
            'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
        ]

    # Датафрейм з датасетом, чи відповідними колонками, які були обрані для візуалізації
    df_display = st.checkbox("Display Raw Data", value=True)
    if df_display:
        st.write(df[cols_to_display] if cols_to_display else df)

    # Функції для створення візуалізацій
    sns.set_style("darkgrid")
    def plot_histogram():
        for col in cols_to_display:
            if df[col].dtype in ['int64', 'float64']:
                st.write(f"**{col}**")
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = sns.color_palette("pastel")
                sns.histplot(df[col], kde=True, ax=ax,
                             color=colors[0], edgecolor='black', linewidth=1.2)
                kde_color = colors[3]
                sns.kdeplot(df[col], ax=ax, color=kde_color, linewidth=2)
                ax.set_title(f"Histogram of {col}",
                             fontsize=18, fontweight='bold')
                ax.set_xlabel(col, fontsize=14)
                ax.set_ylabel("Frequency", fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)

    def plot_boxplot():
        sns.set_style("darkgrid")
        for col in cols_to_display:
            if df[col].dtype in ['int64', 'float64']:
                st.write(f"**{col}**")
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = sns.color_palette("pastel")
                sns.boxplot(x=df[col], ax=ax, color=colors[0], width=0.5, linewidth=2, flierprops={
                            'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
                ax.set_title(f"Boxplot of {col}",
                             fontsize=18, fontweight='bold')
                ax.set_xlabel(col, fontsize=14)
                ax.set_ylabel("Values", fontsize=14)
                ax.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)

    def plot_barplot():
        sns.set_style("darkgrid")
        for col in cols_to_display:
            if df[col].dtype == 'object':
                st.write(f"**{col}**")
                fig, ax = plt.subplots(figsize=(14, 6))
                colors = sns.color_palette("pastel")

                barplot = sns.countplot(x=df[col], ax=ax, palette=colors)
                for p in barplot.patches:
                    height = p.get_height()
                    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=14, fontweight='bold')
                ax.set_title(f"Bar Plot of {col}",
                             fontsize=18, fontweight='bold')
                ax.set_xlabel(col, fontsize=14)
                ax.set_ylabel("Count", fontsize=14)
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

    def plot_piechart():
        for col in cols_to_display:
            if df[col].dtype == 'object':
                st.write(f"**{col}**")
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = sns.color_palette("pastel")
                values = df[col].value_counts()

                def autopct_format(pct):
                    return f'{pct:.1f}%' if pct > 2 else ''
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=values.index,
                    autopct=autopct_format,
                    startangle=140,
                    colors=colors,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                    textprops={'fontsize': 12},
                    pctdistance=0.85,
                    labeldistance=1.1)
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                    autotext.set_color('black')

                ax.set_title(f"Pie Chart of {col}",
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

    def plot_correlation_matrix():
        st.write("#### Correlation Matrix")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm',
                    ax=ax, fmt='.2f', linewidths=0.5)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    if viz_type == "Histogram":
        plot_histogram()
    elif viz_type == "Boxplot":
        plot_boxplot()
    elif viz_type == "Bar Plot":
        plot_barplot()
    elif viz_type == "Pie Chart":
        plot_piechart()
    elif viz_type == "Correlation Matrix":
        plot_correlation_matrix()


with tab2:

    st.write("### 🤖 Analyze Model Predictions")
    st.markdown("""
    This section allows you to explore the performance of a pre-trained machine learning model
    that predicts automobile prices based on their specifications.""")

    if model is not None:
        predictions = predict_model(model, data=df)
        prediction_column = 'prediction_label'
        if prediction_column in predictions.columns:
            df_pred = df.copy()
            df_pred['Prediction'] = predictions[prediction_column]

            # Відображення початку датасету з прогнозами моделі
            st.write("#### 🔍 Data with Predictions")
            st.dataframe(df_pred.head())

            # Розрахунок метрик оцінки моделі
            y_true = df_pred['price']
            y_pred = df_pred['Prediction']
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            med_ae = median_absolute_error(y_true, y_pred)
            
            st.write("#### 📊 Model Metrics")
            st.write(f"📌 **R² (coefficient of determination):** {r2:.4f}")
            st.write(f"📌 **MAE (Mean Absolute Error):** {mae:.2f}")
            st.write(f"📌 **MSE (Mean Squared Error):** {mse:.2f}")
            st.write(f"📌 **RMSE (Root Mean Squared Error):** {rmse:.2f}")
            st.write(f"📌 **Median Absolute Error:** {med_ae:.2f}")

            # Візуалізація порівняння дійсних та прогнозованих значень моделі
            st.write("#### 📊 Predicted vs. Actual Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x=df_pred['price'], y=df_pred['Prediction'], alpha=0.7, color="green")
            ax.plot([df_pred['price'].min(), df_pred['price'].max()],
                    [df_pred['price'].min(), df_pred['price'].max()], '--', color='red')
            ax.set_title("Predicted vs. Actual Values", fontsize=14)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            st.pyplot(fig)

            # Найкращі та найгірші прогнози на основі модулю помилки між прогнозом та дійсним значенням
            st.write("#### 📌 The best predictions")
            df_pred['Abs_Error'] = abs(df_pred['price'] - df_pred['Prediction'])
            st.dataframe(df_pred.nsmallest(5, 'Abs_Error'))
            st.write("#### ❌ The worst predictions")
            st.dataframe(df_pred.nlargest(5, 'Abs_Error'))
            
            # Розподіл прогнозів
            st.write("#### 📊 Distribution of Predicted Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_pred['Prediction'], bins=30,
                         kde=True, color="blue")
            ax.set_title("Distribution of Predicted Values", fontsize=14)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # Гістограма помилок
            st.write("#### 🔍 Error Histogram")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_pred['Error'] = df_pred['price'] - df_pred['Prediction']
            sns.histplot(df_pred['Error'], bins=30, kde=True, color="red")
            ax.set_title("Error Distribution", fontsize=14)
            ax.set_xlabel("Difference between Actual and Predicted Values")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        else:
            st.error(
                "❌ Error: The prediction column is missing in the output of predict_model().")
    else:
        st.error("❌ Model not loaded, please check the file path.")


with tab3:

    st.write("### 🎯 Make Your Own Car Price Prediction")
    st.markdown(
        """Use this interactive tool to **input custom car specifications** and get an instant price estimate!""")

    if model is not None:
        st.markdown("#### 📌 Select Feature Values")
        user_input = {}

        # Розподіл числових та категорій-ознак по двом стовпцям для вибору
        numerical_columns = ['engine-size', 'horsepower', 'curb-weight', 'city-mpg', 'highway-mpg',
                             'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height',
                             'bore', 'stroke', 'compression-ratio', 'peak-rpm']

        categorical_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
                               'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

        def split_list(lst):
            mid = len(lst) // 2
            return lst[:mid], lst[mid:]

        # Елементи для вибору числових значень на основі існуючих мін, макс в датасеті
        def create_sliders(columns, df, col_container):
            for col in columns:
                min_val, max_val, mean_val = float(df[col].min()), float(
                    df[col].max()), float(df[col].mean())
                user_input[col] = col_container.slider(
                    f"Select {col}", min_val, max_val, mean_val)

        # Елементи для вибору значень категорій на основі існуючих унікальних значень в датасеті
        def create_selectboxes(columns, df, col_container):
            for col in columns:
                options = df[col].dropna().unique().tolist()
                user_input[col] = col_container.selectbox(f"Select {col}", options)

        numerical_group1, numerical_group2 = split_list(numerical_columns)
        categorical_group1, categorical_group2 = split_list(
            categorical_columns)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        create_sliders(numerical_group1, df, col1)
        create_sliders(numerical_group2, df, col2)
        create_selectboxes(categorical_group1, df, col3)
        create_selectboxes(categorical_group2, df, col4)

        st.markdown("""
            <style>
                div.stButton > button {
                    width: 100%;
                    background-color: #2c3e50;
                    color: white;
                    font-size: 16px;
                    padding: 15px;
                    border-radius: 5px;
                    border: none;
                }
                div.stButton > button:hover {
                    background-color: #34495e;
                }
            </style>
        """, unsafe_allow_html=True)

        # Виведення результатів прогнозу у вигляді ціни та таблиці з історією прогнозів
        st.session_state.setdefault('prediction_history', pd.DataFrame())
        st.session_state.setdefault('predicted_price', None)
        
        # Щоб зробити прогноз на основі вибраних в елементах вище значень ознак потрібно натиснути на кнопку
        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([user_input])
            prediction_result = predict_model(model, data=input_df)
            predicted_value = prediction_result["prediction_label"].iloc[0]
            st.session_state['predicted_price'] = predicted_value
            user_input['Predicted Price'] = predicted_value
        
            # Зберігати лише 10 останніх прогнозів у таблиці історії
            st.session_state['prediction_history'] = pd.concat(
                [st.session_state['prediction_history'], pd.DataFrame([user_input])], ignore_index=True).iloc[-10:]
        
        # Виведення результату прогнозу у вигляді ціни
        if st.session_state['predicted_price'] is not None:
            st.subheader(f"🔮 Predicted Price: **${st.session_state['predicted_price']:,.2f}**")
        else:
            st.subheader("🔮 Predicted Price")
            st.write("No predicted price yet. Press the button to make a prediction.")
        
        # Виведення історії прогнозів
        if len(st.session_state['prediction_history']) > 0:
            st.write("### 📊 Prediction History Table")
            st.dataframe(st.session_state['prediction_history'])
        else:
            st.write("### 📊 Prediction History Table")
            st.write("No predictions yet. Make a prediction to see history.")
    else:
        st.error("❌ Model not loaded, please check the file path.")

with tab4:
    st.title("🚀 Welcome to the Automobile Analysis & Prediction App!")
    st.write("This application allows you to explore automobile data, visualize key attributes, and predict car prices using machine learning.")

    st.markdown("### 📊 **Visualizations Tab**")
    st.write("""
    - Explore key features of the dataset through histograms, boxplots, bar plots, pie charts, and correlation matrices.
    - Identify trends, relationships, and outliers in vehicle specifications and pricing.
    - Customize visualizations by selecting numerical and categorical attributes of interest.
    """)

    st.markdown("### 🤖 **Model Predictions Tab**")
    st.write("""
    - Utilize a pre-trained machine learning model to estimate the price of a car based on its specifications.
    - View model performance metrics.
    - Analyze prediction accuracy by comparing actual and predicted values.
    """)

    st.markdown("### 🎯 **Interactive Prediction Tab**")
    st.write("""
    - Input custom vehicle specifications (engine size, fuel type, body style, etc.) and get an instant price prediction.
    - Adjust parameters dynamically and observe how different factors impact the price.
    - Keep track of previous predictions with the prediction history table.
    """)

    st.write("Happy exploring! 🎉")