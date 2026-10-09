"""Step 3 - Real-time sign language detection.

Usage:
    python detect_realtime.py

Keys:
    SPACE  add the currently detected sign to the sentence
    BACKSPACE / b  delete last character
    c      clear sentence
    q      quit
"""
import os
import pickle
from collections import Counter, deque

import cv2

from utils import create_hands, detect_hand, draw_hand

MODEL_PATH = os.path.join("model", "sign_model.pkl")
CONF_THRESHOLD = 0.60   # ignore low-confidence predictions
SMOOTH_FRAMES = 10      # majority vote over last N frames (reduces flicker)


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    hands = create_hands()
    history = deque(maxlen=SMOOTH_FRAMES)
    sentence = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        features, lm = detect_hand(frame, hands)
        current, conf = "", 0.0
        if features is not None:
            draw_hand(frame, lm)
            probs = model.predict_proba([features])[0]
            idx = probs.argmax()
            conf = probs[idx]
            if conf >= CONF_THRESHOLD:
                history.append(model.classes_[idx])
        else:
            history.clear()

        if history:
            current = Counter(history).most_common(1)[0][0]

        # ---- overlay ----
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        label_text = f"Sign: {current}  ({conf * 100:.0f}%)" if current else "Show a sign..."
        cv2.putText(frame, label_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 200), 2)
        cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, "Text: " + sentence, (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Sign Language Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and current:
            sentence += current
            history.clear()
        elif key in (8, ord("b")):
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
