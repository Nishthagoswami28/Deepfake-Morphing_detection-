import cv2
import numpy as np

def detect_morph(face):
    if face is None:
        return 0

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Laplacian (detect blur/edges)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Threshold logic (tune later)
    if laplacian_var < 50:
        return 0.8   # likely morph
    else:
        return 0.2   # likely real