# Sign Language Detection using Machine Learning

Real-time hand sign recognition using **MediaPipe Hands** (21 hand landmarks) and a
**Random Forest** classifier, built with Python and OpenCV.

## How it works
Webcam frame -> MediaPipe detects the hand and returns 21 landmarks -> landmarks are
normalised into 42 numbers -> Random Forest predicts the sign -> result is shown on screen.

## Project structure
| File | Purpose |
|------|---------|
| `utils.py` | Hand detection and landmark normalisation helpers |
| `collect_data.py` | Record your own training data from the webcam |
| `train_model.py` | Train the model, print accuracy, save confusion matrix |
| `detect_realtime.py` | Live detection with smoothing and sentence building |
| `requirements.txt` | Python dependencies |

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Use Python 3.9 - 3.11 (MediaPipe does not support the newest Python versions on every OS).

## Run
```bash
python collect_data.py --labels A B C D E --samples 250   # 1. collect data
python train_model.py                                     # 2. train
python detect_realtime.py                                 # 3. live detection
```

## Tips for better accuracy
- Collect 250+ samples per sign; vary hand angle, distance and lighting.
- Keep the same hand and camera setup for collecting and detecting.
- Add more people's hands if possible.

## Limitations
Recognises static signs (letters) only; one hand; accuracy depends on lighting and data.

## Future work
Dynamic signs with LSTM over frame sequences, two-hand support, speech output, mobile app.

## Tech stack
Python, OpenCV, MediaPipe, scikit-learn, NumPy, pandas, Matplotlib
