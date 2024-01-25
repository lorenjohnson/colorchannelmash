import cv2
import numpy as np
import dlib
from skimage import color
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

# cv2.COLORMAP_HOT
def apply_colormap(image, colormap=cv2.COLORMAP_OCEAN):
    return cv2.applyColorMap(image, colormap)

def zoom_image_on_face(image, zoom_percentage=20, preview=False):
    detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) > 0:
        max_face = max(faces, key=lambda rect: rect.width() * rect.height())
        x, y, w, h = max_face.left(), max_face.top(), max_face.width(), max_face.height()
        center_x = x + w // 2
        center_y = y + h // 2
        zoom_factor_x = (zoom_percentage / 100) * w / image.shape[1]
        zoom_factor_y = (zoom_percentage / 100) * h / image.shape[0]
        zoomed_width = int(w * (1 + zoom_factor_x))
        zoomed_height = int(h * (1 + zoom_factor_y))
        roi_x = max(0, center_x - zoomed_width // 2)
        roi_y = max(0, center_y - zoomed_height // 2)
        roi_width = min(image.shape[1], zoomed_width)
        roi_height = min(image.shape[0], zoomed_height)
        roi_x = max(0, center_x - roi_width // 2)
        roi_y = max(0, center_y - roi_height // 2)
        zoomed_roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        zoomed_image = cv2.resize(zoomed_roi, (image.shape[1], image.shape[0]))

        if preview:
            cv2.imshow("Zoomed Face", zoomed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return zoomed_image
    else:
        return image

def randomly_dispose_pixels(image, percentage):
    percentage = max(0.0, min(percentage, 100.0))
    num_pixels = int((percentage / 100.0) * image.size)
    indices = np.arange(image.size)
    np.random.shuffle(indices)
    image_flattened = image.flatten()
    image_flattened[indices[:num_pixels]] = 0
    modified_image = image_flattened.reshape(image.shape)

    return modified_image

def keep_color_channels_separated(image):
    if image.ndim == 3:
        flattened_image = image.reshape((-1, 3))
        num_colors = 8
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(flattened_image)
        quantized_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        normalized_image = quantized_image / 255.0
        quantized_image = img_as_ubyte(normalized_image)

        return quantized_image
    elif image.ndim == 2:
        image[:, :] -= image[:, :]
    else:
        raise ValueError("Unsupported array dimensionality. Only 2D and 3D arrays are supported.")

    image = np.clip(image, 0, 255)

    return image

def keep_color_channels_separated_alt(image):
    if image.ndim == 3:
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]
        blue_offset = np.zeros_like(blue_channel)
        green_offset = np.zeros_like(green_channel)
        red_offset = np.zeros_like(red_channel)
        separated_image = cv2.merge((blue_channel + blue_offset, green_channel + green_offset, red_channel + red_offset))

        return separated_image
    elif image.ndim == 2:
        image[:, :] -= image[:, :]
        return image
    else:
        raise ValueError("Unsupported array dimensionality. Only 2D and 3D arrays are supported.")

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

def reduce_contrast(image):
    channel_min = np.min(image, axis=(0, 1))
    channel_max = np.max(image, axis=(0, 1))
    channel_range = channel_max - channel_min
    contrast_reduction_factor = 50 / (channel_range + 1e-10)
    return np.clip(contrast_reduction_factor * image, 0, 255).astype(np.uint8)

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
