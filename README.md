# Acoustic Keylogger - Proof of Concept

This is a fascinating project that bridges **Digital Signal Processing (DSP)** and **Machine Learning**. It vividly demonstrates how physical side-channels (sound) can leak digital secrets.

Because this is a complex pipeline, we will build a **Proof of Concept (PoC)**. Real-world acoustic keyloggers require very clean data and precise segmentation, but we can build a working prototype that distinguishes between a few distinct keys (e.g., Spacebar vs. 'A' vs. Enter).

## ⚠️ Ethical Disclaimer

**This project is for educational and research purposes only.** Do not use this to record or decode keystrokes from anyone without their explicit consent.

## The Architecture

1. **Data Collection:** Record audio while pressing specific keys to create a "labeled dataset."
2. **Segmentation:** Chop the long audio file into tiny individual clips, each containing exactly one keystroke.
3. **Feature Extraction:** Convert those audio clips into numbers (MFCCs) that a computer understands.
4. **Training:** Teach a classifier to recognize the patterns.

## Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note on pyaudio installation:**
- **macOS:** You may need to install portaudio first: `brew install portaudio`
- **Linux:** `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Windows:** Pre-compiled wheels should work, but if not, try installing from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### Step 3: Run Data Collection

```bash
python 1_record_and_segment.py
```

This script will:
- Prompt you to record each key (Space, Enter, A) 10 times
- Save raw recordings to `raw_data/`
- Automatically segment the audio into individual keystroke clips
- Save segmented clips to `dataset/{KeyName}/`

### Step 4: Train the Model

```bash
python 2_train_model.py
```

This script will:
- Load all segmented audio clips from `dataset/`
- Extract MFCC features from each clip
- Train a Random Forest classifier
- Display accuracy and classification report
- Save the trained model to `models/key_classifier.pkl`

## How to Make It Succeed

1. **Placement Matters:** Place your microphone on the desk surface near the keyboard. The vibrations through the table are often clearer than the sound through the air.
2. **Consistency:** While recording training data, try to type at a consistent speed. In the real world, people type fast (rollover), which blends sounds together. Our segmentation script assumes distinct pauses between keys.
3. **Mechanical Keyboards:** These work best. A loud "Blue" switch mechanical keyboard has very distinct click profiles compared to a mushy laptop membrane keyboard.

## Limitations & Challenges

* **Key Rollover:** If you type "TH" very fast, the sound of T and H overlap. Separating them is a very advanced "Blind Source Separation" problem.
* **Environment Noise:** A cough or a door slamming will look like a keystroke to our simple segmenter.
* **PoC Scope:** This is a proof-of-concept designed to distinguish only a few distinct keys. Real-world keyloggers require much more sophisticated techniques.

## Project Structure

```
type-sensor/
├── 1_record_and_segment.py    # Data collection and segmentation
├── 2_train_model.py           # Feature extraction and ML training
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── raw_data/                  # Raw audio recordings (generated)
├── dataset/                   # Segmented keystroke clips (generated)
│   ├── Space/
│   ├── Enter/
│   └── A/
└── models/                    # Saved trained models (generated)
    └── key_classifier.pkl
```

## Next Steps

- Try running the **Segmentation Script** first to see if we can successfully isolate your individual key presses
- Visualize the MFCCs to see what the "fingerprint" of your Spacebar actually looks like
- Experiment with different keys or different keyboard types
- Try improving the segmentation algorithm for better accuracy

# Keyboard-key-detector
