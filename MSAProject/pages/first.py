import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from streamlit_extras.switch_page_button import switch_page
 
st.set_page_config(initial_sidebar_state='collapsed',layout='wide')
 
st.markdown(
    """
    <center>
        <h1 style="font-size: 3rem;">Data Visualisation Before Cleaning</h1>
    </center>
    """,
    unsafe_allow_html=True,
)

data = pd.read_csv("C:/Users/mehul/Desktop/MSAProject/data/synthetic_business_data.csv")

# Create a 2x2 layout
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# Content for row 1, column 1
with row1_col1:
    st.subheader("Data Summary")
    st.write(data.describe())

# Content for row 1, column 2
with row1_col2:
    st.subheader("Outliers")
    fig = px.box(data, y=['Revenue', 'Expenses', 'Profit'], title="Boxplots of Revenue, Expenses, and Profit")
    st.plotly_chart(fig)

# Content for row 2, column 1
with row2_col1:
    # Visualization using Plotly Express - Scatter Plot
    st.subheader("Visualization: Scatter Plot")
    fig = px.scatter(data, x='Revenue', y='Expenses', color='Category', size=data['Profit'].abs(), hover_data='BusinessName',
                     title='Scatter Plot of Revenue vs. Expenses')
    st.plotly_chart(fig)


# Content for row 2, column 2
with row2_col2:
    # Visualization : Bar Chart - Category Distribution
    st.subheader("Visualization : Bar Chart - Category Distribution")
    fig1 = px.bar(data, x='Category', title='Category Distribution')
    st.plotly_chart(fig1)


# Function to remove null values
def remove_null(data):
    return data.dropna()

# Function to remove garbage values
def remove_garbage(data):
    # Assuming categorical columns contain garbage values
    categorical_columns = ['BusinessName', 'Category', 'City', 'State', 'ZipCode', 'IsOpen', 'Wifi', 'OutdoorSeating', 'Delivery']
    for column in categorical_columns:
        data = data[~data[column].apply(lambda x: isinstance(x, str) and not x.strip())]
    return data

# Function to remove outliers
def remove_outliers(data):
    # Assuming only numerical columns have outliers
    numerical_columns = ['Revenue', 'Expenses', 'Profit', 'ProfitMargin', 'Latitude', 'Longitude']
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


# Clean data button
if st.button('Clean Data'):
    # Remove null values
    data = remove_null(data)

    # Remove garbage values
    data = remove_garbage(data)

    # Remove outliers
    data = remove_outliers(data)

    # Save cleaned data to session state
    st.session_state['cleaned_data'] = data
    
    switch_page("second")

















# # Function to clean the data
# def clean_data(data):
#     # Remove null values
#     data = data.dropna()

#     # Remove outliers (assuming only numerical columns)
#     numerical_columns = data.select_dtypes(include=[np.number]).columns
#     for column in numerical_columns:
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

#     # Remove garbage values (assuming categorical columns)
#     categorical_columns = data.select_dtypes(include=['object']).columns
#     for column in categorical_columns:
#         data = data[~data[column].str.contains(r'\b[A-Za-z]{3,}\b')]

#     return data

# # Add Clean Data Button
# if st.button('Clean Data'):
#     clean_data = clean_data(data)
#     st.write("Data cleaned successfully!")
#     st.write("Cleaned Data:")
#     st.write(clean_data)
#     switch_page("second")


# # # Function to clean data
# # def clean_data(data):
# #     # Placeholder for data cleaning logic
# #     st.write("Data cleaning logic goes here...")

# #     # Data cleaning button
# #     with row2_col2:
# #         st.subheader("Data Cleaning")
# #         if st.button("Clean Data"):
# #             clean_data(data)

