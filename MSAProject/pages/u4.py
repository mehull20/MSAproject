import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(initial_sidebar_state='collapsed',layout='wide')

st.markdown(
    """
    <center>
        <h1 style="font-size: 3rem;">Model Testing</h1>
    </center>
    """,
    unsafe_allow_html=True,
)

# Load the trained models
models = {}
for model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
    model_file = f"{model_name}_model.pkl"
    if os.path.exists(model_file):
        models[model_name] = joblib.load(model_file)

# Function to preprocess the data
def preprocess_data(df):
    le = LabelEncoder()
    categorical_cols = ['IsOpen', 'Wifi', 'OutdoorSeating', 'Delivery']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df


# Model Testing Page
def model_testing_page(selected_model):

        # Load testing data
        df_test = pd.read_csv(uploaded_file)

        # Preprocess testing data
        df_test_processed = preprocess_data(df_test)

        # Load the selected model
        model = models.get(selected_model)

        if model is not None:
            # Make predictions using the selected model
            predictions = model.predict(df_test_processed)

            # Display predictions and inferences
            display_predictions_and_inferences(predictions)
        else:
            st.error("Model not found! Please train the selected model first.")

# Function to display predictions and inferences
def display_predictions_and_inferences(predictions):
    # Count the occurrences of each category in predictions
    prediction_counts = pd.Series(predictions).value_counts()

    # Display the count of each category
    st.subheader("Predicted Category Counts")
    st.write(prediction_counts)

    # For example, you can calculate the percentage distribution of predicted categories
    total_predictions = len(predictions)
    category_percentages = prediction_counts / total_predictions * 100

    # Display the percentage distribution of predicted categories
    st.subheader("Percentage Distribution of Predicted Categories")
    st.write(category_percentages)

col1,col2=st.columns(2)

with col1:
    # Model selection dropdown
    selected_model = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "Gradient Boosting"])

    # Upload testing data file
    uploaded_file = st.file_uploader("Upload Testing Data (CSV format)", type="csv")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

with col2:
    # Display the model testing page
    model_testing_page(selected_model)