# Phishing Email Detector

A powerful machine learning-based application that helps detect phishing emails using an ensemble of Decision Tree and Random Forest classifiers. Built with Streamlit, this application provides both single email analysis and batch processing capabilities.

## Features

- **Single Email Analysis**: Analyze individual emails for phishing attempts
- **Batch Processing**: Upload CSV files for bulk email analysis
- **Real-time Detection**: Instant classification of emails as phishing or legitimate
- **Spam Folder Management**: Automatic organization of detected phishing emails
- **Prediction History**: Track and review all analyzed emails
- **Model Evaluation**: View detailed performance metrics of the detection model

## Technical Details

### Model Architecture
- Ensemble model combining Decision Tree and Random Forest classifiers
- Uses soft voting for final predictions
- Features include:
  - Text analysis (sender, subject, keywords)
  - Suspicious keyword detection
  - Domain reputation checking
  - URL and attachment analysis
  - Time-based features

### Key Features
- Email validation
- Confidence scoring for predictions
- CSV export capabilities
- Interactive visualization of model metrics
- Persistent spam folder storage

## Usage

### Single Email Analysis
1. Enter email details in the sidebar:
   - Sender email address
   - Subject line
   - Keywords
   - URL presence
   - Attachment presence
   - Timestamp
2. Click "predict" to analyze the email
3. View results and confidence scores

### Batch Processing
1. Prepare a CSV file with the following columns:
   - sender
   - subject
   - keywords
   - has_url (optional)
   - has_attachment (optional)
   - timestamp (optional)
2. Upload the CSV file through the interface
3. View batch prediction results

## Requirements

- Python 3.x
- Required Python packages:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - joblib

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run phishing_email_detector.py
```

## Model Training

The application uses a pre-trained model by default. The model is trained on a dataset of labeled emails and includes:
- Text vectorization
- Feature extraction
- Ensemble model training
- Performance evaluation

## Security Features

- Email format validation
- Domain reputation checking
- Suspicious keyword detection
- Time-based anomaly detection

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.

## Authors

- Edwin
- Jael

## Acknowledgments

Special thanks to all contributors and the open-source community for their valuable tools and libraries. 