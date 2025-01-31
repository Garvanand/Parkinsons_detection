import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

url = "data.csv"
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", 
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", 
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", 
            "DFA", "spread1", "spread2", "D2", "PPE", "status"]
dataset = pd.read_csv(url, names=features)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X_scaled, Y, test_size=validation_size, random_state=seed)

clf = DecisionTreeClassifier(random_state=seed, max_depth=5, class_weight='balanced')
clf.fit(X_train, Y_train)

st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🩺", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.write("Explore the app features:")
app_mode = st.sidebar.radio("Choose a section", ["Home", "Upload & Predict", "Model Performance", "About"])

if app_mode == "Home":
    st.title("Parkinson's Disease Prediction App 🎙️")
    st.write("Welcome to the **Parkinson's Disease Prediction App**! This app uses machine learning to analyze audio features and predict the likelihood of Parkinson's Disease. Upload an audio file (WAV or MP3) to get started.")
    st.image("https://via.placeholder.com/800x400.png?text=Parkinson's+Disease+Prediction", use_column_width=True)
    st.write("### How it works:")
    st.write("1. Upload an audio file (e.g., a voice recording).")
    st.write("2. The app extracts audio features and uses a trained model to predict Parkinson's Disease.")
    st.write("3. View the prediction results and model performance metrics.")

elif app_mode == "Upload & Predict":
    st.title("Upload & Predict 🎤")
    st.write("Upload your audio file for analysis and prediction.")

    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.write("### Processing audio and extracting features...")
        sample_features = extract_features(uploaded_file)
        sample_features_scaled = scaler.transform([sample_features])

        prediction = clf.predict(sample_features_scaled)
        prediction_proba = clf.predict_proba(sample_features_scaled)

        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error("Prediction: **Positive for Parkinson's Disease**")
        else:
            st.success("Prediction: **Negative for Parkinson's Disease**")

        st.subheader("Probability of Parkinson's Disease")
        probabilities = pd.DataFrame(prediction_proba, columns=['Negative', 'Positive'])
        st.bar_chart(probabilities.T)

elif app_mode == "Model Performance":
    st.title("Model Performance 📊")
    st.write("Here's how the model performs on the validation set:")

    predictions = clf.predict(X_validation)
    accuracy = accuracy_score(Y_validation, predictions)
    matthews_corr = matthews_corrcoef(Y_validation, predictions)
    report = classification_report(Y_validation, predictions, output_dict=True)
    cm = confusion_matrix(Y_validation, predictions)

    st.subheader("Accuracy")
    st.write(f"**{accuracy * 100:.2f}%**")

    st.subheader("Matthews Correlation Coefficient")
    st.write(f"**{matthews_corr:.2f}**")

    st.subheader("Classification Report")
    st.table(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif app_mode == "About":
    st.title("About 🧠")
    st.write("### Parkinson's Disease Prediction App")
    st.write("This app is designed to predict Parkinson's Disease using voice recordings. It uses a **Decision Tree Classifier** trained on a dataset of voice features. The model extracts **MFCC (Mel-Frequency Cepstral Coefficients)** from audio files and makes predictions based on these features.")
    st.write("### Features:")
    st.write("- Upload and analyze audio files (WAV or MP3).")
    st.write("- View prediction results with probabilities.")
    st.write("- Explore model performance metrics (accuracy, MCC, confusion matrix).")
    st.write("### Built with:")
    st.write("- Python, Streamlit, Scikit-learn, Librosa, and Matplotlib.")
    st.write("### Developer:")
    st.write("Created by [Your Name] as part of a project to explore AI/ML applications in healthcare.")

st.sidebar.markdown("---")
st.sidebar.write("**Disclaimer:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.")
