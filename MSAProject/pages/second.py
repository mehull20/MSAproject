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
        <h1 style="font-size: 3rem;">Data Visualisation After Cleaning</h1>
    </center>
    """,
    unsafe_allow_html=True,
)

# Retrieve cleaned data from session state
cleaned_data = st.session_state.get('cleaned_data')

# Create a 2x2 layout
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# Content for row 1, column 1
with row1_col1:
    st.subheader("Data Summary")
    st.write(cleaned_data.describe())

# Content for row 1, column 2
with row1_col2:
    st.subheader("No Outliers Left")
    fig = px.box(cleaned_data, y=['Revenue', 'Expenses', 'Profit'], title="Boxplots of Revenue, Expenses, and Profit")
    st.plotly_chart(fig)

# Content for row 2, column 1
with row2_col1:
    # Visualization using Plotly Express - Scatter Plot
    st.subheader("Visualization: Scatter Plot")
    fig = px.scatter(cleaned_data, x='Revenue', y='Expenses', color='Category', size=cleaned_data['Profit'].abs(), hover_data='BusinessName',
                     title='Scatter Plot of Revenue vs. Expenses')
    st.plotly_chart(fig)


# Content for row 2, column 2
with row2_col2:
    # Visualization 2: Bar Chart - Category Distribution
    st.subheader("Visualization 2: Bar Chart - Category Distribution")
    fig1 = px.bar(cleaned_data, x='Category', title='Category Distribution')
    st.plotly_chart(fig1)

# Center align a button at the bottom
st.markdown(
    """
    <style>
    .stButton>button {
        margin: 0 auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button('Go for Models'):
    switch_page("third")