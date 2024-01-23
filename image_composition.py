import numpy as np
import cv2

def before(images):
    """
    Check for a single image or blank images and return the appropriate result.
    """
    if len(images) == 1:
        return images[0].copy()  # Combine handle_single_image functionality directly

    non_blank_images = [image for image in images if not (np.all(image == 0) or np.all(image == 255))]
    if not non_blank_images:
        return np.zeros_like(images[0])

    return non_blank_images

def add_image_as_color_channel(target_image, new_image, channel_index):
    """
    Replace a channel in the target image with the corresponding channel from the new image.
    """
    if not np.all(new_image == 0) and not np.all(new_image == 255):
        target_image[:, :, channel_index] = new_image[:, :, channel_index % 3]
    return target_image

def average(images):
    """
    Combine images by taking the average pixel value for each channel.
    """
    processed_images = before(images)
    if len(processed_images) == 1:
        return processed_images[0]
    else:
        return np.mean(processed_images, axis=0).astype(np.uint8)

def overlay(images):
    """
    Combine images using the overlay blending mode.
    """
    processed_images = before(images)
    base_image = processed_images[0].copy() / 255.0
    for overlay_image in processed_images[1:]:
        for channel in range(base_image.shape[-1]):
            base_channel = base_image[:, :, channel]
            overlay_channel = overlay_image[:, :, channel] / 255.0

            base_image[:, :, channel] = (
                2 * base_channel * overlay_channel if base_channel.mean() < 0.5 else 1 - 2 * (1 - base_channel) * (1 - overlay_channel)
            )

    base_image *= 255.0
    return base_image.astype(np.uint8)

def multiply(images):
    """
    Combine images using the multiply blending mode.
    """
    processed_images = before(images)
    result = processed_images[0].copy().astype(np.uint32)
    for overlay_image in processed_images[1:]:
        result = (result * overlay_image.astype(np.uint32)) // 255
    return result.astype(np.uint8)

def screen(images):
    """
    Combine images using the screen blending mode.
    """
    inverted_images = [255 - image for image in before(images)]
    inverted_result = multiply(inverted_images)
    return 255 - inverted_result

def soft_light(images):
    """
    Combine images using the soft light blending mode.
    """
    processed_images = before(images)
    base_image = processed_images[0].copy()
    for overlay_image in processed_images[1:]:
        for channel in range(base_image.shape[-1]):
            base_channel = base_image[:, :, channel] / 255.0
            overlay_channel = overlay_image[:, :, channel] / 255.0

            if overlay_channel.mean() < 0.5:
                result_channel = 2 * base_channel * overlay_channel
            else:
                result_channel = 1 - 2 * (1 - base_channel) * (1 - overlay_channel)

            base_image[:, :, channel] = (result_channel * 255).astype(np.uint8)

    return base_image.astype(np.uint8)