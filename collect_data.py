"""Step 1 - Collect training data from your webcam.

Usage:
    python collect_data.py                       # default signs A B C D E
    python collect_data.py --labels A B C HELLO --samples 300

For every sign: show the sign to the camera, press 's' to start recording.
Move your hand slightly (angle, distance) while it records for better data.
Press 'q' at any time to quit (data collected so far is still saved).
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

from utils import create_hands, detect_hand, draw_hand

DATA_PATH = os.path.join("data", "landmarks.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument("--samples", type=int, default=250, help="samples per sign")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    cap = cv2.VideoCapture(0)
    hands = create_hands()
    rows = []
    quit_all = False

    for label in args.labels:
        if quit_all:
            break

        # ---- wait for the user to get ready ----
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sign: {label}  -  press 's' to start, 'q' to quit",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Collect data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
            if key == ord("q"):
                quit_all = True
                break
        if quit_all:
            break

        # ---- record samples ----
        count = 0
        while count < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            features, lm = detect_hand(frame, hands)
            if features is not None:
                rows.append(np.append(features, label))
                count += 1
                draw_hand(frame, lm)
            cv2.putText(frame, f"{label}: {count}/{args.samples}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Collect data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_all = True
                break
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    if rows:
        cols = [f"f{i}" for i in range(42)] + ["label"]
        df = pd.DataFrame(rows, columns=cols)
        # append to existing data if present
        if os.path.exists(DATA_PATH):
            df = pd.concat([pd.read_csv(DATA_PATH, dtype={"label": str}), df])
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} samples to {DATA_PATH}")
        print(df["label"].value_counts())
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
