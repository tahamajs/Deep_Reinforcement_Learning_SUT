# Author: Taha Majlesi - 810101504, University of Tehran

import cv2
import gymnasium as gym
import numpy as np


def preprocess(frame):
    """Preprocess Atari frames for A3C."""
    frame = frame[34 : 34 + 160, :160]
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame.astype(np.float32)
    frame *= 1.0 / 255.0
    frame = np.expand_dims(frame, 0)
    return frame
