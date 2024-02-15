import os

import numpy as np
import cv2
from PIL import Image
from pillow_lut import load_cube_file

def apply(image, colormap_or_lut):
    # if isinstance(colormap_or_lut, str):
    #     print(f"COLORMAP: {colormap_or_lut}")
    #     return cv2.applyColorMap(image, COLORMAPS[colormap_or_lut])
    # else:
    image = Image.fromarray(image)
    lut = load_cube_file(COLORMAPS[colormap_or_lut])
    image = image.filter(lut)
    return np.array(image)

def load_luts_from_directory(directory='luts'):
    for file_name in os.listdir(directory):
        if file_name.endswith('.cube'):
            lut_name = os.path.splitext(file_name)[0]
            lut_path = os.path.join(directory, file_name)
            # pillow_lut = load_cube_file(lut_path)
            COLORMAPS[lut_name] = lut_path

# List of OpenCV colormaps 
# copilot please make this a dictionary
COLORMAPS = {
#   'autumn': cv2.COLORMAP_AUTUMN,
#   'bone': cv2.COLORMAP_BONE,
#   'jet': cv2.COLORMAP_JET,
#   'winter': cv2.COLORMAP_WINTER,
#   'rainbow': cv2.COLORMAP_RAINBOW,
#   'ocean': cv2.COLORMAP_OCEAN,
#   'summer': cv2.COLORMAP_SUMMER,
#   'spring': cv2.COLORMAP_SPRING,
#   'cool': cv2.COLORMAP_COOL,
#   'hsv': cv2.COLORMAP_HSV,
#   'pink': cv2.COLORMAP_PINK,
#   'hot': cv2.COLORMAP_HOT,
#   'parula': cv2.COLORMAP_PARULA,
#   'magma': cv2.COLORMAP_MAGMA,
#   'inferno': cv2.COLORMAP_INFERNO,
#   'plasma': cv2.COLORMAP_PLASMA,
#   'viridis': cv2.COLORMAP_VIRIDIS,
#   'cividis': cv2.COLORMAP_CIVIDIS,
#   'twilight': cv2.COLORMAP_TWILIGHT,
#   'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
#   'turbo': cv2.COLORMAP_TURBO,
#   'deepgreen': cv2.COLORMAP_DEEPGREEN
}

load_luts_from_directory()
