import pyaudio
import wave
import numpy as np
import os
import librosa
from datetime import datetime

# CONFIGURATION
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 500  # Silence threshold (adjust based on mic sensitivity)
SILENCE_LIMIT = 0.5  # Time in seconds to consider a keystroke "finished"

def record_key(key_label, num_samples=10):
    """Records audio for a specific key multiple times."""
    p = pyaudio.PyAudio()
    
    print(f"\n--- RECORDING KEY: '{key_label}' ---")
    print(f"Please press '{key_label}' {num_samples} times with distinct pauses...")

    # Simple recording logic (fixed duration for simplicity in this PoC)
    # In a pro version, you'd record continuously and detect onsets.
    # Here we record 10 seconds which should fit 10 slow keystrokes.
    DURATION = 10

    print("Get ready to type...")
    print("Recording will start in:")
    print("3...")
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("🎵 RECORDING NOW! START TYPING! 🎵")

    # Open stream AFTER countdown to avoid buffer overflow
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)
        
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the raw full recording
    if not os.path.exists("raw_data"): os.makedirs("raw_data")
    filename = f"raw_data/{key_label}_raw.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def segment_audio(file_path, key_label):
    """Slices the raw audio into individual keystroke files."""
    y, sr = librosa.load(file_path, sr=None)
    
    # Detect Onsets (moments where sound spikes)
    # This is the 'magic' - finding the start of the click
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=10, pre_avg=10, post_avg=10, delta=0.1)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Save clips
    output_dir = f"dataset/{key_label}"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print(f"Detected {len(onset_times)} keystrokes for '{key_label}'. Saving clips...")
    
    for i, start_time in enumerate(onset_times):
        # We take a small window around the onset
        # 50ms before, 250ms after (Typical keystroke duration)
        start_sample = int((start_time - 0.05) * sr)
        end_sample = int((start_time + 0.25) * sr)
        
        # Boundary checks
        if start_sample < 0: start_sample = 0
        if end_sample > len(y): end_sample = len(y)
        
        clip = y[start_sample:end_sample]
        
        # Save individual clip
        clip_name = f"{output_dir}/{key_label}_{i}.wav"
        import soundfile as sf # You might need: pip install soundfile
        sf.write(clip_name, clip, sr)

# --- RUNNER ---
if __name__ == "__main__":
    # Record all alphabetical letters (A-Z), numbers (0-9), and Enter/Space
    import string
    letters = list(string.ascii_uppercase)  # A-Z
    numbers = [str(i) for i in range(10)]    # 0-9
    special_keys = ["Enter", "Space"]

    keys_to_record = letters + numbers + special_keys

    print(f"Will record {len(keys_to_record)} keys: {keys_to_record}")
    print("This will take approximately", len(keys_to_record) * 10, "seconds of recording time")
    input("Press Enter to start recording all keys...")

    for key in keys_to_record:
        raw_file = record_key(key, num_samples=10)
        segment_audio(raw_file, key)

