# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:07:13 2023

@author: hrytsai
"""

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pycaret.classification import *
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
st.title('Робота з візуалізацією даних та Streamlit.')
data = {
        "instant":  "(numeric)",
        "dteday": "date of record",
        "season" : "season of record",
        "yr" : "year of record",
        "mnth": "mnth of record",
        "hr": "hour of record",
        "holiday": "holiday? (binary: yes,no)",
        "weekday": "holiday? (binary: yes,no)"}
index = [i for i in range(len(data))]
df_table = pd.DataFrame(data=data.values(),index=data.keys(),columns=['Explanation'])
st.table(df_table)

df = get_data("bike")
data = df[0:200]
test_data = data[200:]

data = data.rename(columns={"instant": "default_instant",
                            "dteday":"dteday",
                            "season":"season", 
                            "yr": "yr"})
columns = data.columns

cat_data = data.select_dtypes(include=['object'])
num_data = data.select_dtypes(include=['int64'])
num_data["season"].replace({999: 0}, inplace=True)

def univariate_cat_plots(df):
    for feature in df.columns[0:3]:
        plt.figure(figsize=(16,6))
        sns.set_theme(context='notebook',style='darkgrid',palette='deep',color_codes=True)
        
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x=feature,data=df,palette="dark",orient='v')
        plt.xlabel(feature)
        plt.title("Univariate analysis of categorical feature")
        st.pyplot(fig)
        #plt.show()
        
univariate_cat_plots(cat_data)   
le = LabelEncoder()
cat_data= cat_data[cat_data.columns].apply(lambda col: le.fit_transform(col))

bike_data = pd.concat([cat_data,num_data],axis =1)

bike_data = bike_data[columns]
print(bike_data)
bike_data.head()
#%%
bike_data = bike_data.drop(columns=['windspeed'])
#%%
PATH = "./model.pkl"

model = pickle.load(open(PATH, 'rb'))
st.write(model)

st.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name="model.pkl",
)
st.write(model.predict(bike_data))
uploaded_file = st.file_uploader("uploading")

if uploaded_file is not None:
    clf2 = pickle.loads(uploaded_file.read())
    st.write("processing")
    st.write(clf2)
    st.write(test_data.cnt)