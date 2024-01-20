import cv2
import numpy as np
from skimage import color
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

# Apply contrast reduction factor to each channel
def reduce_contrast(frame):
    channel_min = np.min(frame, axis=(0, 1))
    channel_max = np.max(frame, axis=(0, 1))
    channel_range = channel_max - channel_min
    contrast_reduction_factor = 50 / (channel_range + 1e-10)
    return np.clip(contrast_reduction_factor * frame, 0, 255).astype(np.uint8)

def zoom_frame_on_face(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Find the face with the maximum area (most prominent face)
        max_face = max(faces, key=lambda rect: rect[2] * rect[3])

        # Extract coordinates of the most prominent face
        x, y, w, h = max_face

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the required zoom to fill the frame while maintaining the aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_aspect_ratio = w / h

        if target_aspect_ratio > aspect_ratio:
            zoom_factor = frame.shape[1] / w
        else:
            zoom_factor = frame.shape[0] / h

        # Calculate the size of the zoomed-in region
        zoomed_size = (int(w * zoom_factor), int(h * zoom_factor))

        # Calculate the region of interest (ROI) for zooming
        roi_x = max(0, center_x - zoomed_size[0] // 2)
        roi_y = max(0, center_y - zoomed_size[1] // 2)
        roi_width = min(frame.shape[1], zoomed_size[0])
        roi_height = min(frame.shape[0], zoomed_size[1])

        # Extract the ROI from the frame
        zoomed_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Resize the ROI to fill the full frame dimensions
        zoomed_frame = cv2.resize(zoomed_roi, (frame.shape[1], frame.shape[0]))

        cv2.imshow("Face Zoomed", zoomed_frame)
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

# Function to apply the color manipulation effect
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