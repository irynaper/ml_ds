from pycaret import regression

from pycaret.datasets import get_data
import streamlit as st
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px


data = get_data('boston')
model = regression.load_model('best_model')

predicted = regression.predict_model(model, data=data)

st.title('Boston Data Visualization')
st.write('## Columns Info')
st.write('''The `medv` variable is the *target variable*.

Data description - The Boston data frame has 506 rows and 14 columns.



*This data frame contains the following columns:*

* `crim`
per capita crime rate by town.

* `zn`
proportion of residential land zoned for lots over 25,000 sq.ft.

* `indus`
proportion of non-retail business acres per town.

* `chas`
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

* `nox`
nitrogen oxides concentration (parts per 10 million).

* `rm`
average number of rooms per dwelling.

* `age`
proportion of owner-occupied units built prior to 1940.

* `dis`
weighted mean of distances to five Boston employment centres.

* `rad`
index of accessibility to radial highways.

* `tax`
full-value property-tax rate per \$10,000.

* `ptratio`
pupil-teacher ratio by town.

* `black`
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

* `lstat`
lower status of the population (percent).

* `medv`
median value of owner-occupied homes in \$1000s.''')


st.write('## Boston Data')
st.write(data)

st.write('## Country Data Description')
st.write(data.describe())

st.write('## Model')
st.write('* gbr\n* Gradient Boosting Regressor\n* MAE: 2.3014\n* MSE: 11.3751\n* RMSE: 3.3118\n* R2: 0.8610\n* RMSLE: 0.1532\n* MAPE: 0.1196')

st.write('## Predicted Data')
st.write(predicted)

st.write('## Distributions of Variables')

x_variable = st.selectbox('Select X variable', data.columns)
sns.kdeplot(data=data, x=x_variable)
st.pyplot(plt.gcf())

st.write('### Histograms')
fig, ax = plt.subplots()
ax.hist(data[x_variable])
ax.set_xlabel(x_variable)
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.write('## Correlation matrix')
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.5)
st.write(fig)

st.write('## Correlation between two selected attributes')

# Create a selectbox to choose a column to plot on x-axis
x_col = st.selectbox("Select a column to plot on x", data.columns)

# Create a selectbox to choose a column to plot on y-axis
y_col = st.selectbox("Select a column to plot on y", data.columns)

# Create a scatter plot of the selected columns
fig = px.scatter(data, x=x_col, y=y_col)
st.plotly_chart(fig)


st.write('## Residual error')
# calculate residual errors
residuals = predicted['medv'] - predicted['prediction_label']

# Streamlit app to display plot
st.title("Residual Error Plot")
st.write("This plot shows the distribution of residual errors for a linear regression model.")
st.pyplot(sns.displot(residuals, kde=True))