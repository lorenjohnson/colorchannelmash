import argparse
import glob
import cv2
import os
import random
import numpy as np
# import keyboard

# colorchannelmash.py
#
# Utilizes the OpenCV library to create a video montage by combining frames from multiple
# source videos. The resulting video is a composition of randomly selected frames from three
# input videos, each contributing a single color channel to the final frame. The videos are
# resized to a common aspect ratio before being combined.

def parse_args():
    parser = argparse.ArgumentParser(description="Video montage script with command line parameters.")
    parser.add_argument("sourceGlob", nargs='?', default="source/*.(mov|avi|mp4)",
                        help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.(mov|avi|mp4)'.")
    parser.add_argument("--numSets", type=int, default=1, help="Total number of sets to generate. Optional, defaults to 1.")
    parser.add_argument("--setLength", type=int, default=10, help="Duration of each set in seconds. Optional, defaults to 10 seconds.")
    parser.add_argument("--width", type=int, default=1242, help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
    parser.add_argument("--height", type=int, default=2688, help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
    parser.add_argument("--outputDir", default="output", help="Output directory for set files. Optional, defaults to 'output'.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos. Optional, defaults to 30.")
    parser.add_argument("--colorSpace", choices=['hsv', 'hls', 'bgr', 'yuv'], default='bgr',
                        help="Color space for displaying and combining frames. Options: hsv, hls, rgb, yuv. Optional, defaults to bgr.")
    return parser.parse_args()

def resize_frame(frame, target_width, target_height):
    # Resize the frame using OpenCV's resize function
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Create an empty canvas of the target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the position to paste the resized frame in the center of the canvas
    y_offset = (target_height - resized_frame.shape[0]) // 2
    x_offset = (target_width - resized_frame.shape[1]) // 2

    # Paste the resized frame onto the canvas
    canvas[y_offset:y_offset + resized_frame.shape[0], x_offset:x_offset + resized_frame.shape[1]] = resized_frame

    return canvas

def display_video_selection(selected_source, channel_indices, width, height, color_space):
    combined_frame = np.zeros((height, width, 3), dtype=np.uint8)

    current_frame_positions = []

    for i, (source_path, channel_index) in enumerate(zip(selected_source, channel_indices)):
        cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_position = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_position)
        current_frame_positions.append(current_frame_position)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        resized_frame = resize_frame(frame, width, height)
        
        # Check if the color space is BGR
        if color_space.lower() == 'bgr':
            converted_frame = resized_frame
        else:
            # Convert to the specified color space
            converted_frame = cv2.cvtColor(resized_frame, getattr(cv2, f'COLOR_BGR2{color_space.upper()}'))

        combined_frame[:, :, i] = converted_frame[:, :, channel_index]

    # Display the frame in real-time
    print("Press 'Enter' to accept the selection, 'Esc' to pause rendering, or 'Q' to exit.")
    cv2.imshow("Video Selection", combined_frame)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key == 13, current_frame_positions  # Return True if 'Enter' key is pressed, False otherwise

def combine_frames_and_write_video(output_path, selected_source, channel_indices, args, confirmation_frame_positions, color_space):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (args.width, args.height), isColor=True)

    # Calculate the number of frames needed for the desired duration
    frames_per_set = int(args.fps * args.setLength)

    current_frame_positions = confirmation_frame_positions

    for _ in range(frames_per_set):
        combined_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        for i, (source_path, channel_index) in enumerate(zip(selected_source, channel_indices)):
            cap = cv2.VideoCapture(source_path)

            if not cap.isOpened():
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame_positions[i] %= total_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_positions[i])

            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            resized_frame = resize_frame(frame, args.width, args.height)

            if color_space.lower() == 'bgr':
                converted_frame = resized_frame
            else:
                converted_frame = cv2.cvtColor(resized_frame, getattr(cv2, f'COLOR_BGR2{color_space.upper()}'))

            combined_frame[:, :, i] = converted_frame[:, :, channel_index]

            current_frame_positions[i] += 1

        # Display the frame in real-time
        cv2.imshow("Video Rendering", combined_frame)
        key = cv2.waitKey(1)  # Wait for a short period (1 millisecond) to update the display

        if key == 27:  # 'Esc' key
            pause_key = pause_rendering_menu()
            if pause_key == 8:  # 'Backspace' key
                print("Set stopped and video deleted.")
                os.remove(output_path)
                return
            elif pause_key == 27:  # 'Esc' key
                print("Set stopped and video kept.")
                break
            # Continue rendering for any other key

        writer.write(combined_frame)

    writer.release()

def pause_rendering_menu():
    print("Rendering paused. Choose an option:")
    print("1. Stop and delete video (Backspace)")
    print("2. Stop and keep video (Esc)")
    print("3. Continue rendering (Any other key)")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key

def main():
    args = parse_args()

    source_paths = glob.glob(args.sourceGlob)
    if not source_paths:
        print(f"No video files found using the provided sourceGlob: {args.sourceGlob}")
        return

    max_set_number = 0
    existing_sets = glob.glob(os.path.join(args.outputDir, 'set-*.avi'))
    if existing_sets:
        max_set_number = max(int(s.split('-')[1].split('.')[0]) for s in existing_sets)

    for set_number in range(max_set_number + 1, max_set_number + 1 + args.numSets):
        output_path = os.path.join(args.outputDir, f"set-{set_number:03d}.avi")

        while True:
            selected_source = random.sample(source_paths, 3)
            channel_indices = [random.randint(0, 2) for _ in range(3)]

            confirmation_result, confirmation_frame_positions = display_video_selection(selected_source, channel_indices, args.width, args.height, args.colorSpace)

            if confirmation_result:
                print("Set accepted. Generating video set.")
                combine_frames_and_write_video(output_path, selected_source, channel_indices, args, confirmation_frame_positions, args.colorSpace)
                break

if __name__ == "__main__":
    main()
