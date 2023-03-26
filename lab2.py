import streamlit as st
from pycaret import classification
import plotly.express as px

st.write("""
Для лабораторної роботи використовується dataset Titanic.

Опис полів:

Survived Survived (0 = No; 1 = Yes)

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

name Name

sex Sex

age Age

SibSp Number of Siblings/Spouses Aboard

Parch Number of Parents/Children Aboard

ticket Ticket Number

fare Passenger Fare (British pound)

cabin Cabin

embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
""")

model = classification.load_model('final_rf_model')

predictions = classification.predict_model(model)

hist_fig = px.histogram(predictions, x='Survived', nbins=10, 
                        title='Сomparison of the number of survivors and dead people')

st.plotly_chart(hist_fig)