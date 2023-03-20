import plotly.express as px
from pycaret.classification import load_model, predict_model
from pycaret.datasets import get_data
import streamlit as st

# Load the dataset
data = get_data('us_presidential_election_results')

# Load the best model
model = load_model('best_model')

# Select the features
features = ['dem_poll_avg', 'dem_poll_avg_margin']

# Make predictions
predictions = predict_model(model, data[features])

# Create a scatter plot of the predicted probability of the Democratic Party winning vs. the average poll margin
scatter_fig = px.scatter(predictions, x='dem_poll_avg_margin', y='prediction_score',
                         labels={'Score Democratic': 'Predicted Probability of Democratic Party Winning'},
                         title='Scatter Plot of Predicted Probability of Democratic Party Winning vs. Average Poll Margin')

# Create a histogram of the predicted probability of the Democratic Party winning
hist_fig = px.histogram(predictions, x='prediction_score', nbins=20, 
                        labels={'Score Democratic': 'Predicted Probability of Democratic Party Winning'},
                        title='Histogram of Predicted Probability of Democratic Party Winning')

# Show the plots using Streamlit
st.plotly_chart(scatter_fig)
st.plotly_chart(hist_fig)