#%%
import streamlit as st
import pandas as pd
import numpy as np
import pycaret.datasets as cds
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pycaret.classification import *
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

st.title('Робота з візуалізацією даних та Streamlit.')
airQ_df = cds.get_data("airquality")
airQ_df.head()

#%%

airQ_df.columns

#%%

airQ_df['DateTime'] = airQ_df.Date + ' ' + airQ_df.Time
airQ_df['DateTime'] = pd.to_datetime(airQ_df.DateTime)
airQ_df = airQ_df[airQ_df["T"] != -200]
airQ_df.head()

#%%

airQ_df = airQ_df.rename(columns={
    "CO(GT)": "CO_GT",
    "PT08.S1(C0)": "PT08_S1_CO",
    "NMHC(GT)": "NMHC_GT",
    "C6H6(GT)": "C6H6_GT",
    'PT08.S2(NMHC)': "PT08_S2_NMHC",
    'NOx(GT)': "Nox_GT",
    'PT08.S3(NOx)': "PT08_S3_Nox",
    'NO2(GT)': "NO2_GT",
    'PT08.S4(NO2)': "PT08_S4_NO2",
    'PT08.S5(O3)': "PT08_S5_O3"
})


def univariate_cat_plots(df):
    for feature in df.columns[0:3]:
        plt.figure(figsize=(16, 6))
        sns.set_theme(context='notebook', style='darkgrid', palette='deep', color_codes=True)

        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x=feature, data=df, palette="dark", orient='v')
        plt.xlabel(feature)
        plt.title("Univariate analysis of categorical feature")
        st.pyplot(fig)
        # plt.show()

univariate_cat_plots(airQ_df)


regr = linear_model.LinearRegression()

air_x = airQ_df["PT08_S4_NO2"]
air_y = airQ_df["T"]

splits = train_test_split(air_x, air_y, test_size=0.1, random_state=42)

air_x_train, air_x_test, air_y_train, air_y_test = list(map(lambda x: np.array(x).reshape(-1, 1),
                                                            splits))

# Load the model
with open("model.pkl", "rb") as f:
    regr = pickle.load(f)

# Make predictions using the testing set
air_y_pred = regr.predict(air_x_test)

p1 = px.scatter(x=air_x_test.ravel(), y=air_y_test.ravel())
#%%
p2 = px.line(x=air_x_test.ravel(), y=air_y_pred.ravel())

st.plotly_chart(p1)
st.plotly_chart(p2)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(air_y_test, air_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(air_y_test, air_y_pred))

