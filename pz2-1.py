import plotly.express as px
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data
import streamlit as st

# завантажуємо датасет
data = get_data('us_presidential_election_results')

# завантажуємо модель
model = load_model('best_model')

# робимо прогноз
predictions = predict_model(model, data)

# Створюємо гістограмми спрогнозованих верогідностей перемоги на виборах демократів та республіканців
win_prob = predictions[predictions['party_winner'] == "republican"]['prediction_score']
hist_fig = px.histogram(win_prob, nbins=10, labels={'value': 'Probability for Republican', 'count': 'Count'})
st.plotly_chart(hist_fig)

win_prob_dem = predictions[predictions['party_winner'] == "democrat"]['prediction_score']
hist_fig_dem = px.histogram(win_prob_dem, nbins=10, labels={'value': 'Probability for Democrats', 'count': 'Count'})
st.plotly_chart(hist_fig_dem)

# Створюємо кругову діаграму спрогнозованих верогідностей перемоги на виборах демократів та республіканців
pie_fig = px.pie(predictions, names='party_winner')
st.plotly_chart(pie_fig)