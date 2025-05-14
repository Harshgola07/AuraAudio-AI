import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import tempfile
from st_audiorec import st_audiorec
import time

####################################
# Feature Extraction Functions
####################################
def zcr(data, frame_length, hop_length):
    """Extract zero crossing rate."""
    zcr_val = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_val)


def rmse(data, frame_length=2048, hop_length=512):
    """Extract root-mean-square energy."""
    rmse_val = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_val)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    """Extract MFCC features."""
    mfcc_val = librosa.feature.mfcc(y=data, sr=sr)
    # Transpose so that time is along the first axis.
    return np.squeeze(mfcc_val.T) if not flatten else np.ravel(mfcc_val.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """
    Extract features from the audio data by horizontally stacking:
      - Zero Crossing Rate,
      - Root-Mean-Square Energy, and
      - MFCC features.
    """
    result = np.array([])
    result = np.hstack((
        result,
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return result


def get_features(path, duration=2.5, offset=0.6):
    """
    Load an audio file and extract features.
    """
    data, sr = librosa.load(path, duration=duration, offset=offset)
    features = extract_features(data, sr)
    return features


def preprocess_audio_features(file_bytes, duration=2.5, offset=0.6):
    """
    Write the audio bytes to a temporary file, extract features using get_features,
    adjust the feature vector to have a fixed length of 2376 via padding or trimming,
    and reshape it to (1, 1, 2376) for model input.
    """
    try:
        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            temp_path = tmp.name

        # Extract features from the original audio
        features = get_features(temp_path, duration=duration, offset=offset)

        # Ensure the feature vector has the expected length (2376)
        if features.shape[0] != 2376:
            st.warning(f"Expected feature length of 2376, but got {features.shape[0]}. Adjusting via padding/trimming.")
            if features.shape[0] < 2376:
                pad_width = 2376 - features.shape[0]
                features = np.pad(features, (0, pad_width), mode='constant')
            else:
                features = features[:2376]

        # Reshape to (1, 1, 2376)
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return x_tensor
    except Exception as e:
        st.error(f"Error processing audio features: {e}")
        return None


####################################
# Define the EmotionCNN Model
####################################
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, input_length=2376):
        super(EmotionCNN, self).__init__()
        # First convolution block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        # Second convolution block
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.2)

        # Third convolution block
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        # Fourth convolution block
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.2)

        # Fifth convolution block
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(0.2)

        # With padding='same' the final time dimension is assumed to be 75.
        final_length = 75
        self.fc1 = nn.Linear(128 * final_length, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)  # Raw logits
        return x


####################################
# Helper Functions for Model Inference
####################################
def load_model(model_path):
    """
    Load the pre-trained PyTorch model.
    """
    model = EmotionCNN(num_classes=7, input_length=2376)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model


def get_emotion_label(pred):
    """
    Map prediction index to an emotion label.
    Modify these labels as per your training.
    """
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Fearful', 'Disgust', 'Surprised']
    if 0 <= pred < len(emotions):
        return emotions[pred]
    return "Unknown"


def predict_emotion(model, audio_tensor):
    """
    Run inference on the preprocessed audio features and return the predicted emotion.
    """
    with torch.no_grad():
        output = model(audio_tensor)
        pred_index = torch.argmax(output, dim=1).item()
    return get_emotion_label(pred_index)


####################################
# Main Streamlit App
####################################
def main():
    st.title("Real-Time Emotion Detection from Audio")
    st.write("Upload an audio file or record your voice to detect the emotion using our CNN model.")

    # Load model (update model path as needed)
    model = load_model('best_model2.pt')

    # Sidebar for input method selection
    input_method = st.sidebar.radio("Select Input Method", ["Upload Audio File", "Record Audio"])

    if input_method == "Upload Audio File":
        st.header("Upload an Audio File")
        uploaded_file = st.file_uploader("Choose an audio file (wav, mp3, ogg)", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            st.audio(file_bytes, format='audio/wav')
            audio_tensor = preprocess_audio_features(file_bytes)
            if audio_tensor is not None:
                emotion = predict_emotion(model, audio_tensor)
                progress_text = "Operation in progress. Please wait."
                bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                bar.empty()
                st.success(f"Predicted Emotion: **{emotion}**")

    elif input_method == "Record Audio":
        st.header("Record Your Voice")
        st.write("Click the record button below to capture your voice. When finished, the audio will be processed.")

        audio = st_audiorec()
        if audio is not None:
            # Convert the UploadedFile object to bytes
            recorded_bytes = audio
            audio_tensor = preprocess_audio_features(recorded_bytes)
            if audio_tensor is not None:
                emotion = predict_emotion(model, audio_tensor)
                progress_text = "Operation in progress. Please wait."
                bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                bar.empty()

                st.success(f"Predicted Emotion: **{emotion}**")



if __name__ == "__main__":
    main()
