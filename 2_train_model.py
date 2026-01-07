import librosa
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def extract_features(file_path):
    """Loads audio and extracts MFCC features."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Generate MFCCs (Mel-frequency cepstral coefficients)
        # We take the mean to get a single vector per audio clip
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_dir="dataset"):
    features = []
    labels = []
    
    # Iterate through all labels (folders)
    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        if os.path.isdir(path):
            print(f"Processing label: {label}")
            for file in glob.glob(os.path.join(path, "*.wav")):
                data = extract_features(file)
                if data is not None:
                    features.append(data)
                    labels.append(label)
                    
    return np.array(features), np.array(labels)

# --- RUNNER ---
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("ERROR: No samples found in dataset directory!")
        print("Please run 1_record_and_segment.py first to collect data.")
        exit(1)
    
    print(f"Loaded {len(X)} samples.")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Classifier
    # Random Forest is robust and works well with high-dimensional data like MFCCs
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training Random Forest...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- RESULTS ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    if not os.path.exists("models"): os.makedirs("models")
    model_path = "models/key_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {model_path}")

