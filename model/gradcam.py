from pathlib import Path

import cv2
import numpy as np


def generate_heatmap(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    residual = cv2.absdiff(gray, blur)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    residual = residual.astype(np.uint8)

    color_map = cv2.applyColorMap(residual, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, color_map, 0.4, 0)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), overlay)
    return str(output_file).replace("\\", "/")
