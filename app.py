import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from sklearn.model_selection import train_test_split
import librosa

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

st.title("Parkinson's Disease Prediction App")
st.write("Upload your audio file for analysis and prediction.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.write("Processing audio and extracting features...")
    sample_features = extract_features(uploaded_file)
    sample_features_scaled = scaler.transform([sample_features])

    prediction = clf.predict(sample_features_scaled)
    prediction_proba = clf.predict_proba(sample_features_scaled)

    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.write("Prediction: **Positive for Parkinson's Disease**")
    else:
        st.write("Prediction: **Negative for Parkinson's Disease**")

    st.subheader("Probability of Parkinson's Disease")
    probabilities = pd.DataFrame(
        prediction_proba, columns=['Negative', 'Positive']
    )
    st.bar_chart(probabilities.T)

    predictions = clf.predict(X_validation)
    accuracy = accuracy_score(Y_validation, predictions)
    matthews_corr = matthews_corrcoef(Y_validation, predictions)

    st.subheader("Model Performance on Validation Set")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Matthews Correlation Coefficient: {matthews_corr:.2f}")
    st.text(classification_report(Y_validation, predictions))
 