from pycaret import clustering
from pycaret.datasets import get_data
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
data = get_data("nba")
model = clustering.load_model('NBA_1')
st.title('NBA PRED')
st.write('## NBA Info')
st.write("""* Name - Name player
* GP - Games played
* MIN - Minutes per game
* FGM - Field goals made
* FG% - Field goal percentage%
* 3PM - 3-point Field goals made
* 3P% - 3-point field percentage
* FTM - free throws made
* REB - Rebounds per game
* AST - assist per game. """)
x_variable = st.selectbox('Select X variable for displaying distributions', data.drop('Name', axis=1).columns)
sns.kdeplot(data=data, x=x_variable)
st.pyplot(plt.gcf())







