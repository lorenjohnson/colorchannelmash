import dlib
import cv2
import numpy as np
from skimage import color
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

# Other favorites: COLORMAP_SUMMER, COLORMAP_SPRING, COLORMAP_VIRIDIS, COLORMAP_OCEAN
# ref. https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
def apply_colormap(frame, map = cv2.COLORMAP_OCEAN):
    return cv2.applyColorMap(frame, map)

# Apply contrast reduction factor to each channel
def reduce_contrast(frame):
    channel_min = np.min(frame, axis=(0, 1))
    channel_max = np.max(frame, axis=(0, 1))
    channel_range = channel_max - channel_min
    contrast_reduction_factor = 50 / (channel_range + 1e-10)
    return np.clip(contrast_reduction_factor * frame, 0, 255).astype(np.uint8)

def zoom_frame_on_face(frame, zoom_percentage=20, preview = False):
    # Load the frontal face detector and the facial landmarks predictor
    detector = dlib.get_frontal_face_detector()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if len(faces) > 0:
        # Find the face with the maximum area (most prominent face)
        max_face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Extract coordinates of the most prominent face
        x, y, w, h = max_face.left(), max_face.top(), max_face.width(), max_face.height()

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the zoom factor based on a percentage of the face size
        zoom_factor_x = (zoom_percentage / 100) * w / frame.shape[1]
        zoom_factor_y = (zoom_percentage / 100) * h / frame.shape[0]

        # Calculate the size of the zoomed-in region
        zoomed_width = int(w * (1 + zoom_factor_x))
        zoomed_height = int(h * (1 + zoom_factor_y))

        # Calculate the region of interest (ROI) for zooming
        roi_x = max(0, center_x - zoomed_width // 2)
        roi_y = max(0, center_y - zoomed_height // 2)
        roi_width = min(frame.shape[1], zoomed_width)
        roi_height = min(frame.shape[0], zoomed_height)

        # Adjust ROI to ensure it's centered on the face
        roi_x = max(0, center_x - roi_width // 2)
        roi_y = max(0, center_y - roi_height // 2)

        # Extract the ROI from the frame
        zoomed_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Resize the ROI to fill the full frame dimensions
        zoomed_frame = cv2.resize(zoomed_roi, (frame.shape[1], frame.shape[0]))

        if preview:
            cv2.imshow("Head Zoomed", zoomed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return zoomed_frame
    else:
        return frame

def randomly_dispose_pixels(frame, percentage):
    """
    Randomly disposes of N percentage of pixels in a video frame.

    Parameters:
    - frame: The input video frame (NumPy array).
    - percentage: The percentage of pixels to dispose (float between 0 and 100).

    Returns:
    - A new video frame with randomly disposed pixels.
    """
    # Ensure the percentage is within a valid range
    percentage = max(0.0, min(percentage, 100.0))

    # Calculate the number of pixels to dispose
    num_pixels = int((percentage / 100.0) * frame.size)

    # Create an array of indices representing all pixels in the frame
    indices = np.arange(frame.size)

    # Randomly shuffle the indices
    np.random.shuffle(indices)

    # Set the values of the selected indices to black (you can modify this if needed)
    frame_flattened = frame.flatten()
    frame_flattened[indices[:num_pixels]] = 0  # Set to black (0)

    # Reshape the modified array back to the original frame shape
    modified_frame = frame_flattened.reshape(frame.shape)

    return modified_frame

# Function to separate color channels
# Separates color channels in a 3D image using KMeans clustering for quantization,
# and subtracts the blue channel from the green channel in 2D images.
def keep_them_separated(frame):
    if frame.ndim == 3:
        # Reshape the frame to a flat array for KMeans clustering
        flattened_frame = frame.reshape((-1, 3))

        # Perform KMeans clustering for color quantization
        num_colors = 8  # You can adjust the number of colors as needed
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(flattened_frame)
        quantized_frame = kmeans.cluster_centers_[kmeans.labels_].reshape(frame.shape)

        # Normalize the pixel values to the range [0, 1]
        normalized_frame = quantized_frame / 255.0

        # Convert the normalized image to uint8 format
        quantized_frame = img_as_ubyte(normalized_frame)

        return quantized_frame
    elif frame.ndim == 2:
        # 2D array, treat it as a single-channel image
        # Subtract the blue channel from the green channel
        frame[:, :] -= frame[:, :]

    else:
        raise ValueError("Unsupported array dimensionality. Only 2D and 3D arrays are supported.")

    # Clip values to ensure they are within the valid range (0 to 255)
    frame = np.clip(frame, 0, 255)

    return frame

# Function to separate color channels (alternate)
# Creates distinct regions for each color channel in a 3D image by introducing offsets,
# and subtracts the blue channel from the green channel in 2D images.
def keep_them_separated_alt(image):
    if image.ndim == 3:
        # Get individual color channels
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # Create separate images for each channel with offsets
        blue_offset = np.zeros_like(blue_channel)
        green_offset = np.zeros_like(green_channel)
        red_offset = np.zeros_like(red_channel)

        # Combine the separated channels with offsets
        separated_image = cv2.merge((blue_channel + blue_offset, green_channel + green_offset, red_channel + red_offset))

        return separated_image
    elif image.ndim == 2:
        # 2D array, treat it as a single-channel image
        # Subtract the blue channel from the green channel
        image[:, :] -= image[:, :]
        return image
    else:
        raise ValueError("Unsupported array dimensionality. Only 2D and 3D arrays are supported.")