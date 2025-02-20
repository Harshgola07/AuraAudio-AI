# AuraAudio-AI
# Real-Time Emotion Detection App

A Streamlit-based application that detects emotions in real-time from audio input.

## Description

The Real-Time Emotion Detection App processes live or uploaded audio clips to identify and display the underlying emotional state. Utilizing advanced audio processing techniques and machine learning models, the app offers a user-friendly interface for immediate emotion analysis.

## Key Features

- **Real-Time Processing:** Analyze audio input instantly to detect emotions as they occur.
- **Accurate Emotion Detection:** Leverages state-of-the-art machine learning models to provide precise emotion classification.
- **User-Friendly Interface:** Built with Streamlit, the intuitive UI makes it easy for users of all levels.
- **Flexible Input Options:** Supports both live audio recording and file uploads in multiple formats.
- **Adaptive Audio Handling:** If the audio duration is very short, the app compensates through padding and trimming to match the required input size.
- **Customizable Settings:** Adjust parameters to fine-tune the analysis according to your needs.

## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/real-time-emotion-detection-app.git

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv env
   .\env\Scripts\activate

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt

# Usage Instructions

1. **Launch the App:**
   ```bash
   streamlit run app.py
   
2. **Interact with the Interface:**

- Use the provided controls to record live audio or upload an audio file.
- The app will automatically process the input and display the detected emotion.
- If the audio duration is very short, the app will automatically adjust through padding and trimming to ensure consistent input size.

# License

This project is licensed under the Apache-2.0 License.

# Additional Notes

Audio Duration Handling: If the audio clip is shorter than expected, the application will compensate through padding and trimming to match the required input size, ensuring accurate and consistent emotion detection
