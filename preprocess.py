# -----------------------
# 📦 FEATURE EXTRACTION FILE (same as yours)
# -----------------------
import librosa
import numpy as np

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def get_emotion(filename):
    try:
        return emotion_map[filename.split('-')[2]]
    except:
        return None

def extract_features(file_path, extra=True):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs.T, axis=0))

    if extra:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))

    return np.array(features)
