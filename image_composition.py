import numpy as np
import cv2

def handle_single_image(image):
    """
    Shared function to handle the case where only one image is provided.
    """
    return image.copy()

def is_blank(image):
    """
    Check if the image is entirely black or white.
    """
    return np.all(image == 0) or np.all(image == 255)

def average(images):
    """
    Combine images by taking the average pixel value for each channel.
    """
    if len(images) == 1:
        return handle_single_image(images[0])

    images = [image for image in images if not is_blank(image)]
    if not images:
        return np.zeros_like(images[0])

    return np.mean(images, axis=0).astype(np.uint8)

def overlay(images):
    """
    Combine images using the overlay blending mode.
    """
    if len(images) == 1:
        return handle_single_image(images[0])

    base_image = images[0].copy()
    for overlay_image in images[1:]:
        if not is_blank(overlay_image):
            for channel in range(base_image.shape[-1]):
                base_image[:, :, channel] = (
                    2 * base_image[:, :, channel] * overlay_image[:, :, channel] // 255
                    if overlay_image[:, :, channel].mean() < 128
                    else 255 - 2 * (255 - base_image[:, :, channel]) * (255 - overlay_image[:, :, channel]) // 255
                )
    return base_image.astype(np.uint8)

def multiply(images):
    """
    Combine images using the multiply blending mode.
    """
    if len(images) == 1:
        return handle_single_image(images[0])

    result = images[0].copy().astype(np.uint32)
    for overlay_image in images[1:]:
        if not is_blank(overlay_image):
            result = (result * overlay_image.astype(np.uint32)) // 255
    return result.astype(np.uint8)

def screen(images):
    """
    Combine images using the screen blending mode.
    """
    if len(images) == 1:
        return handle_single_image(images[0])

    inverted_images = [255 - image for image in images if not is_blank(image)]
    if not inverted_images:
        return np.zeros_like(images[0])

    inverted_result = multiply(inverted_images)
    return 255 - inverted_result

def soft_light(images):
    """
    Combine images using the soft light blending mode.
    """
    if len(images) == 1:
        return handle_single_image(images[0])

    base_image = images[0].copy()
    for overlay_image in images[1:]:
        if not is_blank(overlay_image):
            for channel in range(base_image.shape[-1]):
                if overlay_image[:, :, channel].mean() < 128:
                    base_image[:, :, channel] = (
                        2 * base_image[:, :, channel] * overlay_image[:, :, channel] // 255
                    )
                else:
                    base_image[:, :, channel] = (
                        255 - 2 * (255 - base_image[:, :, channel]) * (255 - overlay_image[:, :, channel]) // 255
                    )
    return base_image.astype(np.uint8)
