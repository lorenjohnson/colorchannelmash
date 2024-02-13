import cv2

# List of OpenCV colormaps 
# copilot please make this a dictionary
COLORMAPS = {
  'autumn': cv2.COLORMAP_AUTUMN,
  'bone': cv2.COLORMAP_BONE,
  'jet': cv2.COLORMAP_JET,
  'winter': cv2.COLORMAP_WINTER,
  'rainbow': cv2.COLORMAP_RAINBOW,
  'ocean': cv2.COLORMAP_OCEAN,
  'summer': cv2.COLORMAP_SUMMER,
  'spring': cv2.COLORMAP_SPRING,
  'cool': cv2.COLORMAP_COOL,
  'hsv': cv2.COLORMAP_HSV,
  'pink': cv2.COLORMAP_PINK,
  'hot': cv2.COLORMAP_HOT,
  'parula': cv2.COLORMAP_PARULA,
  'magma': cv2.COLORMAP_MAGMA,
  'inferno': cv2.COLORMAP_INFERNO,
  'plasma': cv2.COLORMAP_PLASMA,
  'viridis': cv2.COLORMAP_VIRIDIS,
  'cividis': cv2.COLORMAP_CIVIDIS,
  'twilight': cv2.COLORMAP_TWILIGHT,
  'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED,
  'turbo': cv2.COLORMAP_TURBO,
  'deepgreen': cv2.COLORMAP_DEEPGREEN
}

def apply(image, colormap):
  return cv2.applyColorMap(image, COLORMAPS[colormap])
