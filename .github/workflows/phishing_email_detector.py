import streamlit as st
import base64
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime
import re
import matplotlib.pyplot as plt  # For plotting
import os  # For file checking

# Set Algerian font for the entire app
st.markdown("""
<style>
    * {
        font-family: 'Algerian', sans-serif !important;
        font-size: 12px !important;
    }
    .stTitle, .stHeader, h1, h2, h3, h4, h5, h6, .streamlit-header, [data-testid="stHeader"] {
        font-family: 'Algerian', sans-serif !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebarNav"] .css-17lntkn, [data-testid="stSidebarNav"] h1, 
    .css-10trblm, [data-testid="stHeadingText"], .main .block-container h1, 
    .main .block-container h2, .main .block-container h3 {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Define suspicious keywords and trusted domains
SUSPICIOUS_KEYWORDS = ['urgent', 'win', 'free', 'claim', 'prize', 'now', 'crypto', 'winner', 'security']
TRUSTED_DOMAINS = ['gmail.com', 'uni-pr.edu', 'linkedin.com', 'outlook.com', 'zoom.us', 'github.com', 'dropbox.com', 'trello.com', 'google.com', 'microsoft.com']

# Validation functions
def is_valid_email(email):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def is_valid_subject(subject):
    return isinstance(subject, str) and len(subject.strip()) >= 3

def is_valid_keywords(keywords):
    return isinstance(keywords, str) and len(keywords.strip()) >= 3

def is_valid_timestamp(timestamp):
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$'
    if re.match(timestamp_pattern, timestamp):
        try:
            pd.to_datetime(timestamp)
            return True
        except ValueError:
            return False
    return False

# Preprocess and extract features
def preprocess_data(df, vectorizer=None):
    required_cols = ['sender', 'subject', 'keywords']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''
    
    if 'has_url' not in df.columns:
        df['has_url'] = 0
    if 'has_attachment' not in df.columns:
        df['has_attachment'] = 0
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    df['text'] = df['sender'] + ' ' + df['subject'] + ' ' + df['keywords']
    if vectorizer is None:
        vectorizer = CountVectorizer()
        X_text = vectorizer.fit_transform(df['text'])
    else:
        X_text = vectorizer.transform(df['text'])
    
    df['suspicious_keywords'] = df['subject'].apply(lambda x: sum(word in str(x).lower() for word in SUSPICIOUS_KEYWORDS)) + \
                                df['keywords'].apply(lambda x: sum(word in str(x).lower() for word in SUSPICIOUS_KEYWORDS))
    df['exclamation_count'] = df['subject'].str.count('!') + df['keywords'].str.count('!')
    df['is_suspicious_domain'] = df['sender'].apply(lambda x: 0 if any(domain in str(x) for domain in TRUSTED_DOMAINS) else 1)
    
    df['has_url'] = df['has_url'].fillna(0).astype(int)
    df['has_attachment'] = df['has_attachment'].fillna(0).astype(int)
    df['is_odd_hour'] = df['timestamp'].apply(lambda x: 1 if pd.to_datetime(x, errors='coerce').hour < 9 or 
                                            pd.to_datetime(x, errors='coerce').hour > 17 else 0)
    df['sender_length'] = df['sender'].str.len().fillna(0)
    
    feature_names = vectorizer.get_feature_names_out().tolist() + ['suspicious_keywords', 'exclamation_count', 'is_suspicious_domain', 
                                                                  'has_url', 'has_attachment', 'is_odd_hour', 'sender_length']
    X = pd.concat([pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out()), 
                   df[['suspicious_keywords', 'exclamation_count', 'is_suspicious_domain', 'has_url', 'has_attachment', 'is_odd_hour', 'sender_length']]],
                  axis=1)
    X.columns = feature_names
    y = df['label'].apply(lambda x: 1 if x == 'phishing' else 0) if 'label' in df.columns else None
    
    return X, y, vectorizer, feature_names

# Load or train ensemble model
@st.cache_resource
def load_or_train_model():
    df = pd.read_csv('phishdataset.csv')
    X, y, vectorizer, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('dt', dt_model), ('rf', rf_model)],
        voting='soft'
    )
    
    ensemble_model.fit(X_train, y_train)
    joblib.dump(ensemble_model, 'phishing_ensemble_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    return ensemble_model, vectorizer, X_test, y_test, feature_names

# Load model and components
model, vectorizer, X_test, y_test, feature_names = load_or_train_model()

# Initialize session state for history log and spam folder
if 'history' not in st.session_state:
    st.session_state.history = []
if 'spam_folder' not in st.session_state:
    st.session_state.spam_folder = []  # Start with an empty spam folder

# Streamlit UI
st.markdown("<h1 style='text-align: center; font-family: Algerian;'>PHISHING EMAIL DETECTOR</h1>", unsafe_allow_html=True)

# Sidebar for single email input
st.sidebar.header("INPUT EMAIL DETAILS")
sender = st.sidebar.text_input("Sender Email", placeholder="eg., example@example.com")
subject = st.sidebar.text_input("Subject", placeholder="eg., Test email")
keywords = st.sidebar.text_input("Keywords", placeholder="eg., Key phrases")
has_url = st.sidebar.checkbox("Contains URL")
has_attachment = st.sidebar.checkbox("Has Attachment")
timestamp = st.sidebar.text_input("Timestamp (YYYY-MM-DD HH:MM)", placeholder="eg., 2025-03-24 14:00")

# Process single input
input_df = pd.DataFrame([[sender, subject, keywords, has_url, has_attachment, timestamp]], 
                        columns=['sender', 'subject', 'keywords', 'has_url', 'has_attachment', 'timestamp'])

# Predict button logic
errors = []  # Initialize errors list outside the button
if st.sidebar.button("predict"): 
    # Validation checks
    if not is_valid_email(sender):
        errors.append("Sender email is invalid! Must be a valid email (e.g., example@domain.com).")
    if not is_valid_subject(subject):
        errors.append("Subject is invalid! Must be at least 3 characters long.")
    if not is_valid_keywords(keywords):
        errors.append("Keywords are invalid! Must be at least 3 characters long.")
    if not is_valid_timestamp(timestamp):
        errors.append("Timestamp is invalid! Must be in format YYYY-MM-DD HH:MM (e.g., 2025-03-24 14:00).")

    if errors:
        for error in errors:
            st.sidebar.error(error)
    else:
        X_input, _, _, _ = preprocess_data(input_df, vectorizer)
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][prediction]
        result = "Phishing" if prediction == 1 else "Legitimate"
        
        # Add to history
        email_data = {
            "Sender": sender,
            "Subject": subject,
            "Keywords": keywords,
            "Has URL": "Yes" if has_url else "No",
            "Has Attachment": "Yes" if has_attachment else "No",
            "Timestamp": timestamp,
            "Prediction": result,
            "Confidence": f"{probability:.2f}"
        }
        st.session_state.history.append(email_data)
        
        # Add to spam folder if phishing (only from sidebar) and save to CSV
        if prediction == 1:
            st.session_state.spam_folder.append(email_data)
            pd.DataFrame(st.session_state.spam_folder).to_csv("spam_folder.csv", index=False)
            st.sidebar.warning("Email classified as Phishing and moved to Spam Folder!")
        
        st.write("### Prediction")
        if prediction == 1:
            st.error(f"This email is likely **Phishing**! (Confidence: {probability:.2f})")
        else:
            st.success(f"This email is likely **Legitimate**! (Confidence: {probability:.2f})")

# Upload CSV for batch predictions
st.header("BATCH PREDICTION")
uploaded_file = st.file_uploader("Upload CSV (sender, subject, keywords; optional: has_url, has_attachment, timestamp)", type="csv")
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    required_cols = ['sender', 'subject', 'keywords']
    if all(col in uploaded_df.columns for col in required_cols):
        X_batch, _, _, _ = preprocess_data(uploaded_df, vectorizer)
        predictions = model.predict(X_batch)
        probabilities = model.predict_proba(X_batch)
        uploaded_df['Prediction'] = ['Phishing' if p == 1 else 'Legitimate' for p in predictions]
        uploaded_df['Confidence'] = [max(prob) for prob in probabilities]
        
        # Display only input columns, hiding Prediction and Confidence
        st.write("### BATCH PREDICTION")
        # Create display columns list and capitalize each column name
        display_cols = [col for col in ['sender', 'subject', 'keywords', 'has_url', 'has_attachment', 'timestamp'] 
                        if col in uploaded_df.columns]  # Exclude 'Prediction' and 'Confidence'
        
        # Capitalize column names in the dataframe
        displayed_df = uploaded_df[display_cols].copy()
        displayed_df.columns = [col.upper() for col in displayed_df.columns]
        
        # Display dataset with scrollbar and a descriptive text
        st.write("*SCROLL TO VIEW MORE*:")
        st.dataframe(displayed_df, height=200)
    else:
        st.error("CSV must contain 'sender', 'subject', and 'keywords' columns.")

# History Log
st.header("PREDICTION HISTORY")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.write("No predictions yet.")

# Spam Folder
st.header("SPAM FOLDER")
if st.session_state.spam_folder:
    spam_df = pd.DataFrame(st.session_state.spam_folder)
    st.dataframe(spam_df)
    
    # Clear Spam Folder
    if st.button("Clear Spam Folder"):
        st.session_state.spam_folder = []
        pd.DataFrame(st.session_state.spam_folder).to_csv("spam_folder.csv", index=False)
        st.success("Spam Folder cleared!")
    
    # Download Spam Folder
    csv = spam_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="spam_folder.csv">Download Spam Folder</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.write("No phishing emails in the Spam Folder yet.")

# Model Evaluation
st.header("MODEL EVALUATION")
if st.checkbox("Show Model Metrics"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(metrics.keys(), metrics.values(), width=0.4, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("***© Copyright 2025. Designed by Edwin and Jael.***")
st.markdown("***Current date: 2025-04-06 09:45***")