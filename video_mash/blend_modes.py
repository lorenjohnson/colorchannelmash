import cv2
import numpy as np
import blend_modes as blend_modes_module

def channels_blend_mode(image, new_image, channel_index):
    """
    Replace a channel in the target image with the corresponding channel from the new image.
    """
    if not np.all(new_image == 0) and not np.all(new_image == 255):
        image[:, :, channel_index] = new_image[:, :, channel_index % 3]
    return image

def accumulate_blend_mode(image, new_image, channel_index):
    # Previously only used for "gray": accumulate the intensity values
    image[:, :, channel_index] += new_image[:, :, channel_index]
    return image


def apply(mode_name, provided_image, new_image, layer_index, opacity=0.5):
    """
    Apply a specified blend mode to the provided and new images.

    Args:
        mode_name (str): The name of the blend mode to apply.
        provided_image (numpy.ndarray): The provided image to which the new image will be blended.
        new_image (numpy.ndarray): The new image to be blended onto the provided image.
        layer_index (int): Index of the layer.
        opacity (float, optional): Opacity of the new image in the blend operation (default is 0.5).

    Returns:
        numpy.ndarray: The resulting image after applying the blend mode.
    """
    channel_index = layer_index % 3    

    if provided_image is None:
        # Handle blank initial image
        if mode_name in ['channels', 'multiply']:
            # Black Image
            image = np.zeros_like(new_image)
        elif mode_name in ['add', 'lighten_only']:
            # All White Image
            image = np.ones_like(new_image) * 255
        else:
            # All Gray Image
            image = np.ones_like(new_image) * 128
    else:
        image = provided_image.copy()

    blend_mode_function_name = mode_name + "_blend_mode"
    blend_mode_function = globals().get(blend_mode_function_name)

    if blend_mode_function is not None and callable(blend_mode_function):
        image = blend_mode_function(image, new_image, channel_index)
    elif hasattr(blend_modes_module, mode_name) and callable(getattr(blend_modes_module, mode_name)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image = image.astype(float)

        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
        new_image = new_image.astype(float)

        blend_mode = getattr(blend_modes_module, mode_name)
        image = blend_mode(image, new_image, opacity)
        
        image = image[:, :, :3]
        image = image.astype(np.uint8)
        
        # NOTE: Not necessary to convert new_image back at this time
        # new_image = new_image[:, :, :3]
        # new_image = new_image.astype(np.uint8)
    else:
        print(f"Blend mode function not found for mode: {mode_name}. Images not blended!")

    return image

BLEND_MODES = [
    'channels', 'accumulate'
    # from module
    'soft_light', 'lighten_only', 'dodge', 'addition', 'darken_only', 'multiply', 'hard_light',
    'difference', 'subtract', 'grain_extract', 'grain_merge', 'divide', 'overlay', 'normal'
    ]
