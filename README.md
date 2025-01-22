<<<<<<< HEAD
Multilingual Offensive Detection Using BERT
Project Overview
Objective: Detects offensive content in text, images, and audio across multiple Indian languages.
Classification: Classifies content as offensive, sarcastic, or neutral.
Model: Utilizes BERT for advanced language understanding.
Features
Multimodal Content: Supports text, images, and audio inputs.
Multilingual: Detects offensive language in various Indian languages.
Real-Time Translation: Converts regional languages (e.g., Telugu) to English for analysis.
Dynamic Input: Works with new inputs, not relying on predefined phrases.
Technologies Used
Frontend: HTML, CSS, Bootstrap
Backend: Flask (Python)
Machine Learning: BERT (Bidirectional Encoder Representations from Transformers)
Database: MongoDB (for storing user data)
Version Control: Git and GitHub
Dataset
Uses a labeled dataset for offensive content detection.
Dataset Path: C:\Users\Kaif Md\Downloads\labeled_data.csv.zip
Steps to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Shaik-Mohammad-Kaif/MultilingualOffensiveDetectionUsingBert.git
Navigate to the folder:

bash
Copy
Edit
cd Multilingual_OffenseDetection
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # For Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Project Structure
plaintext
Copy
Edit
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
Future Enhancements
Support more Indian languages for detection.
Improve image and audio detection using advanced models.
Integrate with social media platforms for real-time content moderation.
Enhance user interface for better accessibility.
=======
# MultilingualOffenseDetection
>>>>>>> f3c2ed5c2f852093d14e97b10c8153f235db4863
