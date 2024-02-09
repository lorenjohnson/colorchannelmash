import cv2

def rgb_effect(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def hsv_effect(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
def hsl_effect(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
  
def yuv_effect(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

def gray_effect(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def invert_effect(image):
  image[:, :, :] = 255 - image[:, :, :]
  return image

def jpeg_effect(image, quality=45):
    """
    Apply MJPEG compression to the input image.
    
    Parameters:
        image (numpy.ndarray): Input image in OpenCV format (BGR).
        quality (int): JPEG quality parameter (0-100), where 0 is the lowest quality and 100 is the highest (default: 1).
    
    Returns:
        numpy.ndarray: Compressed image in OpenCV format (BGR).
    """
    # Convert image to JPEG with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, result = cv2.imencode('.jpg', image, encode_param)
    
    # Convert JPEG data back to a image
    result = cv2.imdecode(result, cv2.IMREAD_COLOR)
    
    return result

def ocean_effect(image):
  return cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)

def apply(effect_name, image):
    # Construct the name of the function dynamically
    function_name = effect_name + "_effect"
    # Get the function object using getattr
    effect_function = globals().get(function_name)
    
    # Check if the function exists
    if effect_function is not None and callable(effect_function):
        # Call the function
        image = effect_function(image)
    else:
        print(f"Effect function not found: {effect_name}")

    return image

def construct_effects_list():
    """Constructs the list of available effects dynamically."""
    effect_functions = [func for func in globals().values() if callable(func) and func.__name__.endswith('_effect')]
    effects = [func.__name__[:-len('_effect')] for func in effect_functions]
    print("Constructed effects list:", effects)
    return effects

EFFECTS = construct_effects_list()
print("EFFECTS list:", EFFECTS)

EFFECT_COMBOS = [
    [],
    ['hsl', 'yuv', 'invert'],
    ['hsl', 'rgb'],
    ['invert', 'hsv', 'rgb'],
    ['hsv', 'rgb'],
    ['yuv', 'rgb'],
    ['invert', 'hsl', 'rgb'],
    ['invert', 'yuv', 'rgb'],
    ['invert', 'rgb', 'hsv', 'hsl']
] + [[effect] for effect in EFFECTS]
