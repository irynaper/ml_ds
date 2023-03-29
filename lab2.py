# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:31:43 2023

@author: asus
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
        "age":  "(numeric)",
        "job ": "type of job (categorical: admin.,unknown,unemployed,management,housemaid,entrepreneur,student,blue-collar,self-employed,retired,technician,services)",
        "marital" : "marital status (categorical: married,divorced,single; note: divorced means divorced or widowed)",
        "education" : "(categorical: unknown,secondary,primary,tertiary)",
        "default": "has credit in default? (binary: yes,no)",
        "balance": "average yearly balance, in euros (numeric)",
        "housing": "has housing loan? (binary: yes,no)",
        "loan": "has personal loan? (binary: yes,no)"}
index = [i for i in range(len(data))]
df_table = pd.DataFrame(data=data.values(),index=data.keys(),columns=['Explanation'])
st.table(df_table)

#%%

df = get_data("bank")
data = df[0:33000]
test_data = data[33000:]

# Renaming the columns
data = data.rename(columns={"default": "default_credit",
                            "housing":"housing_loan",
                            "loan":"personal_loan", 
                            "poutcome": "prev_attempt"})
columns = data.columns
test_data = test_data.rename(columns={"default": "default_credit",
                            "housing":"housing_loan",
                            "loan":"personal_loan", 
                            "poutcome": "prev_attempt"})

cat_data = data.select_dtypes(include=['object'])
num_data = data.select_dtypes(include=['int64'])
num_data["pdays"].replace({999: 0}, inplace=True)
print(columns)

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

# Merging both dataframes
bank_data = pd.concat([cat_data,num_data],axis =1)
bank_data = bank_data[columns]
bank_data.head()
#%%
bank_data = bank_data.drop(columns=['deposit'])
#%%
PATH = "./model.pkl"

model = pickle.load(open(PATH, 'rb'))
st.write(model)

st.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name="model.pkl",
)
st.write(model.predict(bank_data))
uploaded_file = st.file_uploader("Upload Model")

# if uploaded_file is not None:
#     clf2 = pickle.loads(uploaded_file.read())
#     st.write("Model loaded")
#     st.write(clf2)
#     st.write("Predicting...")
#     st.write(test_data.deposit)
#     st.write("Done!")