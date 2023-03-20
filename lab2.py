from pycaret import clustering
from pycaret.datasets import get_data
import streamlit as st
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px


data = get_data("country-data")

model = clustering.load_model('kmeans')
data.set_index('country', inplace=True)
predicted = clustering.predict_model(model, data=data)
data.reset_index(inplace=True)
st.title('Country Data Visualization')

st.write('## Columns Info')
st.write("""* country - Name of the country
* child_mort - Death of children under 5 years of age per 1000 live births
* exports - Exports of goods and services per capita. Given as %age of the GDP per capita
* health - Total health spending per capita. Given as %age of GDP per capita
* imports - Imports of goods and services per capita. Given as %age of the GDP per capita
* Income - Net income per person
* Inflation - The measurement of the annual growth rate of the Total GDP
* life_expec - The average number of years a new born child would live if the current mortality patterns are to remain the same
* total_fer - The number of children that would be born to each woman if the current age-fertility rates remain the same.
* gdpp - The GDP per capita. Calculated as the Total GDP divided by the total population. """)

st.write('## Country Data')
st.write(data)


st.write('## Model')
st.write('''* n_clusters: 3 \n* Silhouette: 0.2833\n* Calinski-Harabasz: 66.2348\n* Davies-Bouldin:1.2769\n * Homogeneity: 0\n* Rand Index: 0\n* Completeness: 0 ''')

st.write('## Predicted Data')
st.write(predicted)

data = predicted.reset_index()

st.write('## Distributions')

x_variable = st.selectbox('Select X variable for displaying distributions', data.drop('country', axis=1).columns)
sns.kdeplot(data=data, x=x_variable, hue='Cluster')
st.pyplot(plt.gcf())


st.write("## Pie-chart for Clusters")
st.write(data.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
st.pyplot()
st.write(data.iloc[:, -1].value_counts())


st.write("## Heatmap")
corr = data.drop('country', axis=1).corr()
cmap = sns.diverging_palette(250, 15, s=75, l=40,
                            n=9, center="light", as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap=cmap, mask=mask, center=0)

fig, ax = plt.subplots(figsize=(10, 10))
st.write(sns.heatmap(corr, annot=True, linewidths=0.5, cmap=cmap, mask=mask))
st.pyplot()


st.write('## Line Charts')


x_variable = st.selectbox('Select X variable', data.drop(['country', 'Cluster'], axis=1).columns)
y_variable = st.selectbox('Select Y variable', data.drop(['country', 'Cluster'], axis=1).columns)


chart_data = data[[x_variable, y_variable]]
st.line_chart(chart_data)

st.write('## Scatter plots')

# Create a selectbox to choose a column to plot on x-axis
x_col = st.selectbox("Select a column to plot on x-axis", data.columns)

# Create a selectbox to choose a column to plot on y-axis
y_col = st.selectbox("Select a column to plot on y-axis", data.columns)

# Create a scatter plot of the selected columns
fig = px.scatter(data, x=x_col, y=y_col, hover_name="country", color='Cluster')
st.plotly_chart(fig)





