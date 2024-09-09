import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.switch_page_button import switch_page
import joblib

st.set_page_config(initial_sidebar_state='collapsed',layout='wide')

st.markdown(
    """
    <center>
        <h1 style="font-size: 3rem;">Model Evaluation</h1>
    </center>
    """,
    unsafe_allow_html=True,
)

cleaned_data = st.session_state.get('cleaned_data')

# Split data into train and test sets
def split_data(cleaned_data):
    X = cleaned_data.drop(columns=['Category', 'BusinessName', 'City', 'State', 'ZipCode', 'Latitude', 'Longitude'])
    y = cleaned_data['Category']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data (label encoding)
def preprocess_data(X_train, X_test):
    le = LabelEncoder()
    categorical_cols = ['IsOpen', 'Wifi', 'OutdoorSeating', 'Delivery']
    
    # Apply label encoding to categorical columns
    for col in categorical_cols:
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    return X_train, X_test



# Train model
def train_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Invalid model name")
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{model_name}_model.pkl")

    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    return accuracy, precision, recall, auc_roc

# Plot ROC curve
def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)
    n_classes = y_proba.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)



col1,col2=st.columns(2)
 
with col1:
    st.subheader('Model Selection')
    st.write('Choose a model and click on "Train" to train the model.')

    # Load data
    df = cleaned_data

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Model selection dropdown in sidebar
    model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree","Gradient Boosting"])
 
with col2:
    # Train model button
    if st.button("Train"):
        st.write(f"Training {model_name}...")
        model = train_model(model_name, X_train, y_train)
        st.write("Training completed.")

        # Evaluate model
        st.subheader('Model Performance on Validation Data')
        accuracy, precision, recall, auc_roc = evaluate_model(model, X_test, y_test)
        st.write('Accuracy:', accuracy)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.write('AUC-ROC Score:', auc_roc)

        # Plot ROC curve
        st.subheader('ROC Curve')
        plot_roc_curve(model, X_test, y_test)

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

if st.button('Model Testing'):
    switch_page("u4")

