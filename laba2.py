import streamlit as st
import seaborn as sns
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import pycaret
from pycaret.regression import *
from pycaret.datasets import get_data

st.title('Lab #2 Krasnyuk Anton')

st.write("""
For this Lab we will use a dataset based on a case study called "Sarah Gets a Diamond". This case was presented in the first year decision analysis course at Darden School of Business (University of Virginia). The basis for the data is a case regarding a hopeless romantic MBA student choosing the right diamond for his bride-to-be, Sarah. The data contains 6000 records for training. Short descriptions of each column are as follows:

ID: Uniquely identifies each observation (diamond)

Carat Weight: The weight of the diamond in metric carats. One carat is equal to 0.2 grams, roughly the same weight as a paperclip

Cut: One of five values indicating the cut of the diamond in the following order of desirability (Signature-Ideal, Ideal, Very Good, Good, Fair)

Color: One of six values indicating the diamond's color in the following order of desirability (D, E, F - Colorless, G, H, I - Near colorless)

Clarity: One of seven values indicating the diamond's clarity in the following order of desirability (F - Flawless, IF - Internally Flawless, VVS1 or VVS2 - Very, Very Slightly Included, or VS1 or VS2 - Very Slightly Included, SI1 - Slightly Included)

Polish: One of four values indicating the diamond's polish (ID - Ideal, EX - Excellent, VG - Very Good, G - Good)

Symmetry: One of four values indicating the diamond's symmetry (ID - Ideal, EX - Excellent, VG - Very Good, G - Good)

Report: One of of two values "AGSL" or "GIA" indicating which grading agency reported the qualities of the diamond qualities

Price: The amount in USD that the diamond is valued Target Column

Dataset Acknowledgement:
This case was prepared by Greg Mills (MBA ’07) under the supervision of Phillip E. Pfeifer, Alumni Research Professor of Business Administration. Copyright (c) 2007 by the University of Virginia Darden School Foundation, Charlottesville, VA. All rights reserved.




""")

dataset = get_data('diamond')
st.subheader('Diamond dataset')
st.write(dataset)
st.write("The description of  the dataset")
st.write(dataset.describe())

st.write("In this section let us explore our data!")

st.write("Co relation co efficient and p value of carat v/s price", scipy.stats.pearsonr(dataset['Carat Weight'], dataset['Price']))
st.title('Scatterplot between carat and price')
fig1 = plt.figure()
sns.scatterplot(x = dataset['Carat Weight'],y = dataset['Price'])
st.pyplot(fig1)

st.title('Correlation between features')
fig = plt.figure()
ax = sns.heatmap(dataset.corr())
st.pyplot(fig)
    

st.subheader("Conclusion!")
st.write('Carat has a linear relation with price so it can be a good input for our linear regression model, has a low p value and a high corelation and can be used as input feature for our model')

st.sidebar.header('Select what to display')
pol_parties = dataset['Cut'].unique().tolist()
pol_party_selected = st.sidebar.multiselect('Different Cuts', pol_parties, pol_parties)
nb_deputies = dataset['Cut'].value_counts()
nb_mbrs = st.sidebar.slider("Number of members", int(nb_deputies.min()), int(nb_deputies.max()), (int(nb_deputies.min()), int(nb_deputies.max())), 1)

#creates masks from the sidebar selection widgets
mask_pol_par = dataset['Cut'].isin(pol_party_selected)
#get the parties with a number of members in the range of nb_mbrs
mask_mbrs = dataset['Cut'].value_counts().between(nb_mbrs[0], nb_mbrs[1]).to_frame()
mask_mbrs= mask_mbrs[mask_mbrs['Cut'] == 1].index.to_list()
mask_mbrs= dataset['Cut'].isin(mask_mbrs)

df_dep_filtered = dataset[mask_pol_par & mask_mbrs]
st.write(df_dep_filtered)

st.title('Histograms and Regression plot')
fig, saxis = plt.subplots(2, 2,figsize=(16,12))

sns.regplot(x = 'Carat Weight', y = 'Price', data=dataset, ax = saxis[0,0])

# Order the plots from worst to best
sns.barplot(x = 'Cut', y = 'Price', order=['Fair','Good','Very Good','Ideal'], data=dataset, ax = saxis[0,1])
sns.barplot(x = 'Color', y = 'Price', order=['I','H','G','F','E','D'], data=dataset, ax = saxis[1,0])
sns.barplot(x = 'Clarity', y = 'Price', order=['SI1','VS2','VS1','VVS2','VVS1','IF'], data=dataset, ax = saxis[1,1])
st.pyplot(fig)

saved_final_dt = load_model('Final DT Model Nure')
st.success('Model trained successfully')
data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
exp_reg101 = setup(data = data, target = 'Price', session_id=123, normalize = True, experiment_name = 'diamond2')
predict_model(saved_final_dt)
unseen_predictions = predict_model(saved_final_dt, data=data_unseen)
st.title("Prediction results")
st.write(unseen_predictions.head())

fig123 = plt.figure(figsize=(10,8))
plot_model(saved_final_dt, plot='error', display_format='streamlit')
