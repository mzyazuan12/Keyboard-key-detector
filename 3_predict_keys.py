import pyaudio
import numpy as np
import librosa
import pickle
import time
import threading
from queue import Queue

# CONFIGURATION
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 500  # Silence threshold (adjust based on mic sensitivity)
SILENCE_LIMIT = 0.5  # Time in seconds to consider a keystroke "finished"

class KeyPredictor:
    def __init__(self, model_path="models/key_classifier.pkl"):
        # Load the trained model
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        print("Model loaded successfully!")

        self.audio_queue = Queue()
        self.prediction_queue = Queue()
        self.is_listening = False

    def extract_features(self, audio_data):
        """Extract MFCC features from audio data."""
        try:
            # Convert bytes to numpy array
            y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            y = y / 32768.0  # Normalize to -1 to 1

            # Generate MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def detect_keystroke(self, audio_chunk):
        """Check if audio chunk contains a keystroke sound."""
        try:
            y = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            # Simple energy-based detection
            # Avoid NaN issues by ensuring we have valid data
            if len(y) == 0:
                return False

            # Calculate RMS energy (Root Mean Square)
            energy = np.sqrt(np.mean(y**2))

            # Handle edge cases
            if np.isnan(energy) or np.isinf(energy):
                return False

            return energy > THRESHOLD
        except:
            return False

    def prediction_worker(self):
        """Worker thread that processes audio chunks and makes predictions."""
        buffer = []
        silence_counter = 0

        while self.is_listening or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()

                if self.detect_keystroke(chunk):
                    buffer.append(chunk)
                    silence_counter = 0
                else:
                    silence_counter += 1

                # If we've had silence for a while and have buffered audio, process it
                if silence_counter > int(SILENCE_LIMIT * RATE / CHUNK) and buffer:
                    # Combine buffered chunks
                    combined_audio = b''.join(buffer)

                    # Extract features and predict
                    features = self.extract_features(combined_audio)
                    if features is not None:
                        prediction = self.clf.predict([features])[0]
                        confidence = max(self.clf.predict_proba([features])[0])

                        self.prediction_queue.put((prediction, confidence))

                    # Reset buffer
                    buffer = []
                    silence_counter = 0

            time.sleep(0.01)  # Small delay to prevent busy waiting

    def start_listening(self):
        """Start real-time keystroke detection."""
        print("🎵 Starting real-time keystroke detection...")
        print("Type on your keyboard and watch the predictions!")
        print("Press Ctrl+C to stop.")

        self.is_listening = True

        # Start prediction worker thread
        prediction_thread = threading.Thread(target=self.prediction_worker)
        prediction_thread.daemon = True
        prediction_thread.start()

        # Setup audio stream
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       stream_callback=self.audio_callback)

        stream.start_stream()

        try:
            while True:
                if not self.prediction_queue.empty():
                    prediction, confidence = self.prediction_queue.get()
                    print(f"🎹 Detected: {prediction} (confidence: {confidence:.1f})")
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
            self.is_listening = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            prediction_thread.join(timeout=1)
            print("Stopped.")

def main():
    predictor = KeyPredictor()
    predictor.start_listening()

if __name__ == "__main__":
    main()
