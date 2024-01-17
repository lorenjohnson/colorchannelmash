from __init__ import __version__
from typing import Dict
from pathlib import Path
import os
import subprocess
import glob
import random
import argparse
import shlex
import numpy as np
import ffmpeg
import cv2

# Video montage generation tool written in Python. Combines frames from three different sources to create composite videos.
# Customize parameters like source videos, output directory, set duration, number of sets, video dimensions, color space,
# and frames per second using command line arguments.

# Uses OpenCV for video processing. Provides real-time preview during frame selection. Generated video sets are saved in the
# specified output directory with filenames like "set-001.avi", "set-002.avi," etc.

# Includes functionality to pause rendering with options to stop and delete the video or stop and keep the video. 

# Usage:
# python colorchannelmash.py [<sourceGlob>] [--outputDir OUTPUTDIR] [--setLength SETLENGTH] [--numSets NUMSETS]
#                             [--width WIDTH] [--height HEIGHT] [--colorSpace {hsv,hls,rgb,yuv}] [--fps FPS]

# Example:
# python colorchannelmash.py source/*.mov --outputDir output --setLength 10 --numSets 5 --width 1242 --height 2688
#                               --colorSpace hsv --fps 30

def parse_args():
    parser = argparse.ArgumentParser(description="Video montage script with command line parameters.")
    parser.add_argument("sourceGlob", nargs='?', default="source/*.(mov|avi|mp4)",
                        help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.(mov|avi|mp4)'.")
    parser.add_argument("--outputDir", default="output", help="Output directory for set files. Optional, defaults to 'output'.")
    parser.add_argument("--setLength", type=int, default=10, help="Duration of each set in seconds. Optional, defaults to 10 seconds.")
    parser.add_argument("--numSets", type=int, default=1, help="Total number of sets to generate. Optional, defaults to 1.")
    parser.add_argument("--width", type=int, default=1242, help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
    parser.add_argument("--height", type=int, default=2688, help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
    parser.add_argument("--colorSpace", choices=['rgb', 'hsv', 'hls', 'yuv', 'gray'], default='rgb',
                        help="Color space for displaying and combining frames. Options: hsv, hls, rgb, yuv, gray. Optional, defaults to rgb.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos. Optional, defaults to 30.")
    return parser.parse_args()

def resize_and_crop_frame(frame, target_height, target_width, rotate_fit=False):
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Check if rotating the image provides a better fit
        if rotate_fit:
            rotated_frame = cv2.transpose(frame)
            rotated_height, rotated_width = rotated_frame.shape[:2]

            if (rotated_width >= target_height) and (rotated_height >= target_width):
                frame = rotated_frame
                height, width = rotated_height, rotated_width

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new dimensions to fill the target size
        fill_width = target_width
        fill_height = int(fill_width / aspect_ratio)

        if fill_height < target_height:
            fill_height = target_height
            fill_width = int(fill_height * aspect_ratio)

        # Resize the frame maintaining aspect ratio
        resized_frame = cv2.resize(frame, (fill_width, fill_height))

        # Calculate the cropping region
        start_x = max(0, (fill_width - target_width) // 2)
        start_y = max(0, (fill_height - target_height) // 2)
        end_x = min(fill_width, start_x + target_width)
        end_y = min(fill_height, start_y + target_height)

        # Crop the frame to the target size
        cropped_frame = resized_frame[start_y:end_y, start_x:end_x]

        # Create a blank frame of the target size
        result_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Place the cropped frame in the center of the blank frame
        result_frame[:end_y-start_y, :end_x-start_x] = cropped_frame

        return result_frame

    except Exception as e:
        # Handle errors by returning a blank frame
        print(f"Error: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

def add_metadata(video: Path, meta: Dict[str, str], overwrite: bool = True):
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.") from e

    save_path = video.with_suffix('.metadata' + video.suffix)

    metadata_args = []
    for k, v in meta.items():
        metadata_args.extend(['-metadata', f'{k}={v}'])

    args = [
        'ffmpeg',
        '-v', 'quiet',
        '-i', shlex.quote(str(video.absolute())),
        '-movflags', 'use_metadata_tags',
        '-map_metadata', '0',
        *metadata_args,
        '-c', 'copy',
        shlex.quote(str(save_path))
    ]

    if overwrite:
        args.append('-y')

    try:
        # Run ffmpeg command
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace the original file with the new one
        os.replace(save_path, video)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running ffmpeg: {e.stderr.decode()}") from e
    finally:
        # Delete the save file if it still exists
        if os.path.exists(save_path):
            os.remove(save_path)

class ExitException(Exception):
    pass

def combine_frames_and_write_video(output_path, source_paths, source_channel_indices, source_starting_frames, args):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Determine whether the video is color or grayscale based on the chosen color space
    is_color = args.colorSpace.lower() not in ['gray']

    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (args.width, args.height), isColor=is_color)

    # Calculate the number of frames needed for the desired duration
    frames_per_set = int(args.fps * args.setLength)

    current_frame_positions = source_starting_frames.copy()

    print("(Esc) Stop and delete video, keep generating sets")
    print("(k) Stop and keep video")

    for _ in range(frames_per_set):
        combined_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8) if is_color else np.zeros((args.height, args.width), dtype=np.uint8)

        for i, (source_path, channel_index) in enumerate(zip(source_paths, source_channel_indices)):
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

            resized_frame = resize_and_crop_frame(frame, args.height, args.width)

            if is_color:
                if args.colorSpace.lower() not in ['rgb', 'bgr']:
                    converted_frame = cv2.cvtColor(resized_frame, getattr(cv2, f'COLOR_BGR2{args.colorSpace.upper()}'))
                else:
                    converted_frame = resized_frame
                # Swap each source frame into different color channels
                combined_frame[:, :, i % 3] = resized_frame[:, :, 0]
                # Old version
                # combined_frame[:, :, i] = converted_frame[:, :, channel_index]
            else:
                # Calculate dynamic range of pixel values in the grayscale channel
                channel_min = np.min(resized_frame[:, :, channel_index])
                channel_max = np.max(resized_frame[:, :, channel_index])
                channel_range = channel_max - channel_min

                # Adjust the contrast reduction factor based on the dynamic range
                contrast_reduction_factor = 50 / (channel_range + 1e-10)  # Adding a small value to avoid division by zero

                # Reduce contrast on the grayscale channel before adding to the combined frame
                combined_frame[:, :] += np.clip(contrast_reduction_factor * resized_frame[:, :, channel_index], 0, 255).astype(np.uint8)

            current_frame_positions[i] += 1

        cv2.imshow("Video Rendering", combined_frame)
        # Wait for a short period (1 millisecond) to update the display
        key = cv2.waitKey(1)

        if key ==  ord('d'):
            print("Generation stopped", end="")
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f", {output_path} discarded")
            else:
                print("")
            return False
        elif key == ord('k'):
            print(f"Generation stopped, {output_path} saved.")
            break
        elif key == 27: # `Esc` key
            pause_key = pause_rendering_menu()
            if pause_key in [ord('d'), 27]:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"{output_path} discarded")
                raise ExitException
            elif pause_key == ord('k'):
                raise ExitException
        # Continue rendering for any other key

        writer.write(combined_frame)
    writer.release()
    return True

def pause_rendering_menu():
    print("Rendering paused. Choose an option:")
    print("(d | Esc) Delete current video and Exit")
    print("(k) Exit Keep current video and Exit")
    print("(any key) Continue")
    key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return key

def main():
    args = parse_args()

    source_paths = glob.glob(args.sourceGlob)

    if not source_paths:
        print(f"No video files found using the provided sourceGlob: {args.sourceGlob}")
        return

    max_set_number = 0
    existing_sets = glob.glob(os.path.join(args.outputDir, 'set-*'))
    if existing_sets:
        max_set_number = max(int(s.split('-')[1].split('.')[0]) for s in existing_sets)

    set_number = max_set_number + 1

    try:
        while set_number <= max_set_number + args.numSets:
            output_path = os.path.join(args.outputDir, f"set-{set_number:03d}.mp4")

            # Randomly select source videos and channel indices
            # TODO: Add option to prompt user with a imview of the first frame of a randomly selected source
            # which they can choose to use or not, in which case it will present them with another randomly 
            # selected source. Do this for all 3 files. It should only work with way if a '--selectSources'
            # CLI param is true (false by default), otherwise it does what it does now and selects 3 sources
            # at random without prompting.
            selected_source_paths = random.sample(source_paths, 3)
            source_channel_indices = [random.randint(0, 2) for _ in range(3)]

            # Get random initial frame positions for each source video
            source_starting_frames = [random.randint(0, int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT)) - 1) for src in selected_source_paths]

            # Generate video set without confirmation
            print(f"Generating video set {set_number}...")
            success = combine_frames_and_write_video(
                output_path,
                selected_source_paths,
                source_channel_indices,
                source_starting_frames,
                args
            )

            # Save metadata for the video set
            if success:
                set_number += 1
                absolute_source_paths = [os.path.abspath(path) for path in selected_source_paths]
                metadata = {
                    "source_paths": ",".join(absolute_source_paths),
                    "channel_indices": ",".join(map(str, source_channel_indices)),
                    "starting_frames": ",".join(map(str, source_starting_frames)),
                    "run_params": vars(args),
                    "script_version": __version__ if '__version__' in globals() else "Unknown"
                }
                add_metadata(Path(output_path), metadata)
    except ExitException as e:
        if e:
            print(f"{e}")
        print("Bye!")

if __name__ == "__main__":
    main()
