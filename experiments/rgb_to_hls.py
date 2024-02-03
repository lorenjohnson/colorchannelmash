import cv2
import numpy as np
import random
import osxphotos
# import glob
# import sys

def load_image_from_file(file):
    image = cv2.imread(file)
    return image

def load_image_from_files(image_files):
    images = []
    for file in image_files:
        # Load and convert images to HSL
        images.append(load_image_from_file(file))        
    return images

def process_source_images(images):
    images  = [resize_and_crop(image) for image in images]
    return images

def composite_hsl(images):
    # images = [cv2.cvtColor(image, cv2.COLOR_RGB2HLS) for image in images]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2HLS) for image in images]

    # Composite images in HSL color space
    result = np.zeros_like(images[0], dtype=np.float32)
    for hsl_img in images:
        result += hsl_img / len(images)

    # Convert the result back to RGB
    # result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_HLS2RGB)
    # result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_YUV2BGR)
    # result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_HLS2BGR)

    return result

def composite_rgb_and_convert_to_hsl(images):
    # Composite images in RGB color space
    result_rgb = np.zeros_like(images[0], dtype=np.float32)
    for rgb_img in images:
        result_rgb += rgb_img / len(images)

    # Convert the result to HSL
    result_hsl = cv2.cvtColor(result_rgb.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result_hsl

def resize_and_crop(image, width = 1920, height = 1080, rotate_fit=False):
    try:
        target_height = height
        target_width = width

        # image = image_utils.zoom_image_on_face(image)
        # Get image dimensions
        height, width = image.shape[:2]

        # # Check if a person is present
        # if image_utils.is_person_present(image):
        #     # Adjust cropping region to move the center 25% up
        #     center_shift = int(0.8 * target_height)
        # else:
        center_shift = 0

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new dimensions to fill the target size
        fill_width = target_width
        fill_height = int(fill_width / aspect_ratio)

        if fill_height < target_height:
            fill_height = target_height
            fill_width = int(fill_height * aspect_ratio)

        # Resize the image maintaining aspect ratio
        resized_image = cv2.resize(image, (fill_width, fill_height))

        # Check if rotating the image would result in more coverage
        if rotate_fit and fill_height < target_height:
            fill_width, fill_height = fill_height, fill_width

        # Calculate the cropping region
        start_x = max(0, (fill_width - target_width) // 2)
        start_y = max(0, (fill_height - target_height) // 2 - center_shift)
        end_x = min(fill_width, start_x + target_width)
        end_y = min(fill_height, start_y + target_height)

        # Crop the image to the target size
        cropped_image = resized_image[start_y:end_y, start_x:end_x]

        # Create a blank image of the target size
        result_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Place the cropped image in the center of the blank image
        result_image[:end_y-start_y, :end_x-start_x] = cropped_image

        return result_image
    except Exception as e:
        print(f"Error in resize_and_crop: {e}")
        return None

def filter_jpeg_paths(paths):
    return [path for path in paths if path.lower().endswith((".jpg", ".jpeg"))]

def get_apple_photos_images(dbfile = None):
    if dbfile:
        photosdb = osxphotos.PhotosDB(dbfile=dbfile)
    else:
        photosdb = osxphotos.PhotosDB()

    images = photosdb.photos(images=True, movies=False)
    
    # Filter out None results
    images = [image for image in images if image is not None]
    paths = []
    
    for image in images:
        path = None
        if image.hasadjustments:
            path = image.path_edited
        else:
            path = image.path
        if (path is None):
            print(f"ismissing: {image.ismissing}, isreference: {image.isreference}, hidden: {image.hidden}, shared: {image.shared}")
        else:
            paths.append(path)
    paths = filter_jpeg_paths(paths)
    return paths

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <glob_pattern1> <glob_pattern2> ... <glob_patternN>")
    #     sys.exit(1)

    # Get glob patterns from command-line arguments
    # image_files_globs = sys.argv[1:]

    # Accumulate files from all specified paths
    # all_files = []
    # for glob_pattern in image_files_globs:
    #     all_files.extend(glob.glob(glob_pattern))

    all_files = get_apple_photos_images()
    print(all_files)
    # Randomly pick 3 images
    selected_files = random.sample(all_files, min(3, len(all_files)))
    selected_images = load_image_from_files(selected_files)
    selected_images = process_source_images(selected_images)

    # Composite in HSL and convert back to RGB
    result_hsl_composite = composite_hsl(selected_images)

    # Composite in RGB and convert to HSL
    result_rgb_composite_hsl = composite_rgb_and_convert_to_hsl(selected_images)

    # Display the results
    cv2.imshow('HSL Composite (composite_hsl)', result_hsl_composite)
    cv2.imshow('RGB Composite and Convert to HSL (result_rgb_composite_hsl)', result_rgb_composite_hsl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
