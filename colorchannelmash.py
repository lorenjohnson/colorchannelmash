import cv2
import os
import random

def combine_frames_and_write_video(output_path, source_paths, fps, width, height, duration_seconds=60):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Calculate the number of frames needed for the desired duration
    frames_needed = int(fps * duration_seconds)

    for _ in range(frames_needed):
        # Randomly select a source video, frame, and color channel
        source_path = random.choice(source_paths)
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

        # Randomly shuffle the color channels
        channels = [frame[:, :, i] for i in range(3)]
        random.shuffle(channels)

        # Combine the shuffled channels
        combined_frame = cv2.merge(channels)
        writer.write(combined_frame)

    writer.release()

def main():
    source_directory = '../videopage/source'
    output_path = 'output_video.avi'

    video_paths = [os.path.join(source_directory, file) for file in os.listdir(source_directory) if file.endswith(('.mov', '.avi'))]

    if not video_paths:
        print("No video files found in the 'sources' directory.")
        return

    # Choose the video with the highest resolution as the reference for width and height
    max_width = 0
    max_height = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

        cap.release()

    fps = 30  # Modify as needed

    combine_frames_and_write_video(output_path, video_paths, fps, max_width, max_height)

if __name__ == "__main__":
    main()
