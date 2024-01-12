import cv2
import os
import random
import numpy as np
from osxphotos import PhotosDB

def resize_frame(frame, target_width, target_height):
    current_height, current_width = frame.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = current_width / float(current_height)

    # Calculate new dimensions to maintain the aspect ratio
    if current_width > current_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create an empty canvas of the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the position to paste the resized frame in the center of the canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Paste the resized frame onto the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    return canvas

def combine_frames_and_write_video(output_path, source_paths, fps, width, height, duration_seconds=60):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Calculate the number of frames needed for the desired duration
    frames_needed = int(fps * duration_seconds)

    for _ in range(frames_needed):
        # Randomly select three source videos
        selected_sources = random.sample(source_paths, 3)

        # Initialize an empty frame
        combined_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for i, source_path in enumerate(selected_sources):
            if not os.path.exists(source_path):
                print(f"File not found: {source_path}")
                continue

            cap = cv2.VideoCapture(source_path)

            if not cap.isOpened():
                print(f"Error: Could not open video {source_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = random.randint(0, max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"Error reading frame from {source_path}")
                continue

            # Resize the frame while maintaining the aspect ratio
            resized_frame = resize_frame(frame, width, height)

            # Randomly select a color channel and assign it to the combined frame
            channel_index = random.randint(0, 2)
            combined_frame[:, :, i] = resized_frame[:, :, channel_index]

        writer.write(combined_frame)

    writer.release()

def main():
    # Replace 'path_to_your_photos_library' with the path to your Photos library
    photosdb = PhotosDB(osxphotos.utils.get_last_library_path())

    # Get all original video file paths from the Photos library
    source_paths = [photo.original_filename for photo in photosdb.photos(images=False, movies=True)]

    # Filter out non-existing files
    source_paths = [path for path in source_paths if os.path.exists(path)]

    if not source_paths:
        print("No existing original video files found in the Photos library.")
        return

    output_path = 'output_video.avi'
    fps = 30  # Modify as needed

    # Choose the video with the highest resolution as the reference for width and height
    max_width = 0
    max_height = 0
    for source_path in source_paths:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {source_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

        cap.release()

    combine_frames_and_write_video(output_path, source_paths, fps, max_width, max_height)

if __name__ == "__main__":
    main()
