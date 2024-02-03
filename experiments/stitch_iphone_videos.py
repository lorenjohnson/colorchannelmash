import cv2
import glob
import os
import numpy as np

def create_black_frame(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def stitch_videos(video_directory, clips_wide=3, release_interval=10):
    video_files = glob.glob(os.path.join(video_directory, "*.mp4"))

    if not video_files:
        print(f"No video files found in the specified directory: {video_directory}")
        return

    valid_clips = [cv2.VideoCapture(file) for file in video_files if cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_WIDTH) == 1242]

    if not valid_clips:
        print(f"No valid video files found in the specified directory with the expected width of 1242 pixels.")
        return

    output_height = valid_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
    output_width = clips_wide * int(valid_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter("output.mp4", fourcc, 24.0, (output_width, int(output_height)))

    frame_count = 0
    current_frames = [clip.read()[1] for clip in valid_clips]

    while any(frame is not None for frame in current_frames):
        frames = [frame for frame in current_frames if frame is not None]

        # Ensure the number of clips displayed is limited to the specified clips_wide
        frames = frames[:clips_wide]

        frame_concat = cv2.hconcat(frames)

        out.write(frame_concat)
        cv2.imshow('Preview', frame_concat)
        cv2.waitKey(1)  # Add a small delay to display the preview

        frame_count += 1

        current_frames = []

        for i, clip in enumerate(valid_clips):
            ret, frame = clip.read()
            if ret:
                current_frames.append(frame)
            else:
                current_frames.append(None)

    out.release()
    cv2.destroyAllWindows()

    for clip in valid_clips:
        clip.release()

if __name__ == "__main__":
    video_directory = "gen2"  # Replace with your directory path
    clips_wide = 4  # Change this to 4 if you want 4 clips side by side

    stitch_videos(video_directory, clips_wide)
