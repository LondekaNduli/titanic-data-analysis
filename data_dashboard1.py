import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df= pd.read_csv(" https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=column_names)

# Display the raw data
print(df.head())

#Show the Average Sepal Length for Each Species
st.subheader("Average Sepal Length for Each Species")
average_sepal_length = df.groupby('species')['sepal_length'].mean()
st.write(average_sepal_length)

# Display a Scatter Plot Comparing Two Features
st.subheader("Scatter Plot - Sepal Length vs Sepal Width")
fig1, ax1 = plt.subplots()
ax1.scatter(df['sepal_length'], df['sepal_width'], c='blue', alpha=0.5)
ax1.set_title('Scatter Plot of Sepal Length vs Sepal Width')
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
st.pyplot(fig1)

#Filter Data Based on Species
st.subheader("Filter Data by Species")
selected_species = st.selectbox("Select a species to filter", df['species'].unique())
filtered_df = df[df['species'] == selected_species]
st.dataframe(filtered_df)

# Display a Pairplot for the Selected Species
st.subheader(f"Pairplot for {selected_species}")
fig2 = sns.pairplot(filtered_df, hue='species')
st.pyplot(fig2)

#Show the Distribution of a Selected Feature
st.subheader("Distribution of a Selected Feature")
selected_feature = st.selectbox("Select a feature to display its distribution", df.columns[:-1])
fig3, ax3 = plt.subplots()
sns.histplot(df[selected_feature], kde=True, bins=20, ax=ax3)
ax3.set_title(f'Distribution of {selected_feature}')
st.pyplot(fig3)

