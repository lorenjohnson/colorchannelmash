import inspect
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
        if mode_name in ['channels']:
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

def _generate_blend_modes_list():
    # Get a list of all functions that end with '_effect' from the current frame
    # blend_modes_members = inspect.getmembers(inspect.currentframe().f_back)
    # blend_mode_names = [name[:-len('_blend_mode')] for name, obj in blend_modes_members if inspect.isfunction(obj) and name.endswith('_blend_mode')]

    # Get a list of all attributes (including functions) of the blend_modes module
    # all_blend_modes_lib_attributes = dir(blend_modes_module.blending_functions)
    # blend_modes_module_function_names = [attr for attr in all_blend_modes_lib_attributes if callable(getattr(blend_modes_module.blending_functions, attr)) and not attr.startswith('_')]

    blend_mode_names = [
        'channels',
        'accumulate'
    ]
    blend_modes_from_module = [
        'soft_light', 'lighten_only', 'dodge', 'addition', 'darken_only', 'multiply', 'hard_light',
        'difference', 'subtract', 'grain_extract', 'grain_merge', 'divide', 'overlay', 'normal'
    ]

    return blend_mode_names + blend_modes_from_module

BLEND_MODES = _generate_blend_modes_list()




# # DEPRECATED: Using blend_modes library for these instead
# def before(images):
#     """
#     Check for a single image or blank images and return the appropriate result.
#     """
#     if len(images) == 1:
#         return images[0].copy()

#     non_blank_images = [image for image in images if not (np.all(image == 0) or np.all(image == 255))]
#     if not non_blank_images:
#         return np.zeros_like(images[0], dtype=np.uint8)

#     return non_blank_images

# def average(images):
#     """
#     Combine images by taking the average pixel value for each channel.
#     """
#     processed_images = before(images)
#     if len(processed_images) == 1:
#         return processed_images[0]
#     else:
#         return np.mean(processed_images, axis=0).astype(np.uint8)

# def overlay(images):
#     """
#     Combine images using the overlay blending mode.
#     """
#     processed_images = before(images)
#     base_image = processed_images[0].copy() / 255.0
#     for overlay_image in processed_images[1:]:
#         for channel in range(base_image.shape[-1]):
#             base_channel = base_image[:, :, channel]
#             overlay_channel = overlay_image[:, :, channel] / 255.0

#             base_image[:, :, channel] = (
#                 2 * base_channel * overlay_channel if base_channel.mean() < 0.5 else 1 - 2 * (1 - base_channel) * (1 - overlay_channel)
#             )

#     base_image *= 255.0
#     return base_image.astype(np.uint8)

# def multiply(images):
#     """
#     Combine images using the multiply blending mode.
#     """
#     processed_images = before(images)
#     result = processed_images[0].copy().astype(np.uint32)
#     for overlay_image in processed_images[1:]:
#         result = (result * overlay_image.astype(np.uint32)) // 255
#     return result.astype(np.uint8)

# def screen(images):
#     """
#     Combine images using the screen blending mode.
#     """
#     inverted_images = [255 - image for image in before(images)]
#     inverted_result = multiply(inverted_images)
#     return 255 - inverted_result

# def soft_light(images):
#     """
#     Combine images using the soft light blending mode.
#     """
#     processed_images = before(images)
#     base_image = processed_images[0].copy()
#     for overlay_image in processed_images[1:]:
#         for channel in range(base_image.shape[-1]):
#             base_channel = base_image[:, :, channel] / 255.0
#             overlay_channel = overlay_image[:, :, channel] / 255.0

#             if overlay_channel.mean() < 0.5:
#                 result_channel = 2 * base_channel * overlay_channel
#             else:
#                 result_channel = 1 - 2 * (1 - base_channel) * (1 - overlay_channel)

#             base_image[:, :, channel] = (result_channel * 255).astype(np.uint8)

#     return base_image.astype(np.uint8)