import cv2
import numpy as np

def calculate_brightness(image, color_space='BGR'):
    converted_image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_space}2GRAY'))
    max_pixel_value = np.iinfo(converted_image.dtype).max
    brightness = np.mean(converted_image) / max_pixel_value
    return brightness

def calculate_contrast(image, color_space='BGR'):
    converted_image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_space}2GRAY'))
    max_pixel_value = np.iinfo(converted_image.dtype).max
    contrast = np.std(converted_image) / max_pixel_value
    return contrast

def adjust_brightness(image, target_brightness, color_space='BGR'):
    max_pixel_value = np.iinfo(image.dtype).max
    gray_image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_space}2GRAY'))
    current_brightness = calculate_brightness(image, color_space=color_space)
    brightness_ratio = target_brightness / current_brightness

    # Correct for potential inversion
    if brightness_ratio > 1:
        adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=int(max_pixel_value * (brightness_ratio - 1)))
    else:
        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)

    # Clip pixel values to the valid range [0, max_pixel_value]
    adjusted_image = np.clip(adjusted_image, 0, max_pixel_value)

    return adjusted_image

def adjust_contrast(image, target_contrast, color_space='BGR'):
    max_pixel_value = np.iinfo(image.dtype).max
    gray_image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_space}2GRAY'))
    current_contrast = calculate_contrast(image, color_space=color_space)

    # Check for zero current_contrast to avoid division by zero
    if current_contrast == 0:
        adjusted_image = image.copy()
    else:
        contrast_ratio = target_contrast / current_contrast

        # Adjust the image using cv2.convertScaleAbs
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_ratio, beta=0)

        # Clip pixel values to the valid range [0, max_pixel_value]
        adjusted_image = np.clip(adjusted_image, 0, max_pixel_value)

    return adjusted_image

def is_person_present(image, threshold = 0.1):
    # Convert the image to grayscale for simplicity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using the Canny detector
    edges = cv2.Canny(gray, 50, 150)

    # Calculate the percentage of white pixels in the top region of the image
    height, width = edges.shape
    top_region = edges[:height//2, :]
    white_pixel_percentage = np.sum(top_region == 255) / (width * height // 2)

    # Heuristic: If a significant percentage of the top region has edges (white pixels),
    # consider it likely that there's a person in the video
    return white_pixel_percentage > threshold  # You can adjust this threshold based on your observations
