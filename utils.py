"""Shared helpers: MediaPipe hand tracking + landmark normalisation."""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hands(static=False):
    """Create a MediaPipe Hands detector (one hand only)."""
    return mp_hands.Hands(
        static_image_mode=static,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def extract_features(hand_landmarks):
    """Convert 21 hand landmarks into a 42-value feature vector.

    Steps:
      1. Take (x, y) of every landmark.
      2. Subtract the wrist point (landmark 0) -> position independent.
      3. Divide by the largest absolute value -> size/distance independent.
    """
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    max_val = np.abs(pts).max()
    if max_val > 0:
        pts /= max_val
    return pts.flatten()


def detect_hand(frame_bgr, hands):
    """Return (features, landmarks) for the first hand in frame, or (None, None)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None
    lm = result.multi_hand_landmarks[0]
    return extract_features(lm), lm


def draw_hand(frame_bgr, landmarks):
    mp_draw.draw_landmarks(frame_bgr, landmarks, mp_hands.HAND_CONNECTIONS)
