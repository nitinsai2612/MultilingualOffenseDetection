# Multilingual Offensive Detection Using BERT

## Project Overview
- **Objective**: Detects offensive content in text, images, and audio across multiple Indian languages.
- **Classification**: Classifies content as offensive, sarcastic, or neutral.
- **Model**: Utilizes BERT for advanced language understanding.

## Features
- **Multimodal Content**: Supports text, images, and audio inputs.
- **Multilingual**: Detects offensive language in various Indian languages.
- **Real-Time Translation**: Converts regional languages (e.g., Telugu) to English for analysis.
- **Dynamic Input**: Works with new inputs, not relying on predefined phrases.

## Technologies Used
- **Frontend**: HTML, CSS, Bootstrap
- **Backend**: Flask (Python)
- **Machine Learning**: BERT (Bidirectional Encoder Representations from Transformers)
- **Database**: MongoDB (for storing user data)
- **Version Control**: Git and GitHub

## Dataset
- **Description**: Uses a labeled dataset for offensive content detection.

## Steps to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/nitinsai2612/MultilingualOffenseDetection.git
   ```
2. **Navigate to the folder**:
   ```bash
   cd Multilingual_OffenseDetection
   ```
3. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # For Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the app**:
   ```bash
   python app.py
   ```

## Project Structure
```
Multilingual_OffenseDetection/
├── static/                # Contains CSS and JavaScript files
├── templates/             # HTML files for the interface
│   ├── login.html
│   ├── new.html
│   ├── output.html
│   ├── profile.html
│   ├── signup.html
│   └── translated.html
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── labeled_data.csv.zip   # Dataset for offensive content detection
└── README.md              # Project documentation
