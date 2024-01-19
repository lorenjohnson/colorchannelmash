import cv2
import numpy as np
from skimage import color
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

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