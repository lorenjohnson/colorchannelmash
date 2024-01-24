from __init__ import __version__
from typing import Dict
from pathlib import Path
from alive_progress import alive_bar
import os
import subprocess
import glob
import random
import argparse
import shlex
import numpy as np
import cv2
from video_source import VideoSource
import image_utils
import image_composition

def process_source_frame(frame, target_height, target_width, layer_index, color_space):
    frame = resize_and_crop_frame(frame, target_height, target_width)
    # frame = image_utils.keep_color_channels_separated_alt(frame)
    # frame = image_utils.apply_colormap(frame, cv2.COLORMAP_HOT)
    return frame

def process_combined_frame(provided_combined_frame, new_frame, layer_index, args):
    channel_index = layer_index % 3
    is_color = args.colorSpace.lower() not in ['gray']
    
    if provided_combined_frame is None:
        combined_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    else:
        combined_frame = provided_combined_frame.copy()

    if is_color:
        # combined_frame = image_composition.add_image_as_color_channel(combined_frame, new_frame, channel_index)
        combined_frame = image_composition.multiply([combined_frame, new_frame])

        # Invert channels for HLS and YUV (usually a more useful result)
        # if args.colorSpace.lower() in ['hls', 'yuv']:
        #     combined_frame[:, :, :] = 255 - combined_frame[:, :, :]
        if args.colorSpace.lower() not in ['rgb', 'bgr']:
            combined_frame = cv2.cvtColor(combined_frame, getattr(cv2, f'COLOR_BGR2{args.colorSpace.upper()}'))
    else:
        # For grayscale, accumulate the intensity values
        combined_frame[:, :, channel_index] += new_frame[:, :, channel_index]
        # Convert to grayscale color image
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2GRAY)
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2BGR)
    # combined_frame = image_utils.apply_colormap(combined_frame, cv2.COLORMAP_HOT)
    return combined_frame

class ExitException(Exception):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Video montage script with command line parameters.")
    parser.add_argument("sourceGlob", nargs='*', default=["source/*.mov"],
                        help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.mov'.")
    parser.add_argument("--outputDir", default="output", help="Output directory for set files. Optional, defaults to 'output'.")
    parser.add_argument("--setLength", type=int, default=10, help="Duration of each set in seconds. Optional, defaults to 10 seconds.")
    parser.add_argument("--numSets", type=int, default=1, help="Total number of sets to generate. Optional, defaults to 1.")
    parser.add_argument("--width", type=int, default=1242, help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
    parser.add_argument("--height", type=int, default=2688, help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
    parser.add_argument("--colorSpace", choices=['rgb', 'hsv', 'hls', 'yuv', 'gray'], default='rgb',
                        help="Color space for displaying and combining frames. Options: hsv, hls, rgb, yuv, gray. Optional, defaults to rgb.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos. Optional, defaults to 30.")
    return parser.parse_args()

def main():
    args = parse_args()

    source_paths = []
    for source_glob in args.sourceGlob:
        source_paths.extend(glob.glob(source_glob))

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
            selected_sources = select_sources(source_paths, args)

            # Generate video set without confirmation
            print(f"Generating video set {set_number}...")

            success = combine_frames_and_write_video(
                output_path,
                selected_sources,
                args
            )

            # Save metadata for the video set
            if success:
                set_number += 1
                absolute_source_paths = [os.path.abspath(video_source.source_path) for video_source in selected_sources]
                selected_starting_frames = [video_source.starting_frame for video_source in selected_sources]
                metadata = {
                    "source_paths": ",".join(absolute_source_paths),
                    "starting_frames": ",".join(map(str, selected_starting_frames)),
                    "run_params": vars(args),
                    "script_version": __version__ if '__version__' in globals() else "Unknown"
                }
                add_metadata(Path(output_path), metadata)
    
    except ExitException as e:
        cv2.destroyAllWindows()
        if e:
            print(f"{e}")
        print("Bye!")

    finally:
        # Release VideoReader instances when done
        for video_source in selected_sources:
            video_source.release()

def select_sources(source_paths, args):
    selected_sources = []
    selected_starting_frames = []
    combined_frame = None
    preview_frame = None

    layer_index = 0
    while True:
        selected_source = random.choice(source_paths)
        selected_start_frame = random.randint(0, int(cv2.VideoCapture(selected_source).get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

        video_source = VideoSource(selected_source, selected_start_frame)
        processed_frame = get_and_process_frame(video_source, layer_index, args)
        if processed_frame is None:
            video_source.release()
            continue

        preview_frame = process_combined_frame(combined_frame, processed_frame, layer_index, args)
        layer_index += 1
        cv2.imshow(f"Layer {layer_index + 1}: (Enter) Accept | (Esc) Cancel | (any key) Next", preview_frame)

        # Wait for a key press in the display window
        choice = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if choice in [ord(' '), ord('n')]:
            combined_frame = preview_frame.copy()
            selected_sources.append(video_source)
            continue
        elif choice == 13:  # Enter key
            selected_sources.append(video_source)
            break
        elif choice == 27:  # `Esc` key
            video_source.release()
            raise ExitException
        # "s" saves the 
        else:
            video_source.release()

    return selected_sources

def get_and_process_frame(video_source, layer_index, args):
    frame = video_source.get_frame()
    if frame is None:
        return None

    # Apply any processing to the frame
    frame = process_source_frame(frame, args.height, args.width, layer_index, args.colorSpace)
    return frame

def combine_frames_and_write_video(output_path, video_sources, args):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    is_color = args.colorSpace.lower() not in ['gray']
    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (args.width, args.height), isColor=is_color)

    # Calculate the number of frames needed for the desired duration
    frames_per_set = int(args.fps * args.setLength)

    print("(Esc) Stop and delete video, keep generating sets")
    print("(k) Stop and keep video")

    with alive_bar(frames_per_set) as bar:
        for _ in range(frames_per_set):
            combined_frame = None

            for index, video_source in enumerate(video_sources):
                # video_source.get_frame()
                processed_frame = get_and_process_frame(video_source, index, args)
                if processed_frame is None:
                    continue

                combined_frame = process_combined_frame(combined_frame, processed_frame, index, args)
                video_source.starting_frame += 1

            writer.write(combined_frame)
            bar()

            if not combined_frame is None:

                preview_result = preview_frame(combined_frame, output_path)
                if preview_result == True:
                    break
                elif preview_result == False:
                    return False

    writer.release()
    cv2.destroyAllWindows()

    return True

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
        # '-map_metadata', '0',
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

def preview_frame(frame, output_path):
    rendering_title = "(Esc) Pause | (d) Stop and Delete | (k) Stop and Keep"
    cv2.imshow(rendering_title, frame)
    key = cv2.waitKey(1)

    if key ==  ord('d'):
        print("Generation stopped", end="")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f", {output_path} discarded")
        cv2.destroyAllWindows()
        return False
    elif key == ord('k'):
        print(f"Generation stopped, {output_path} saved.")
        cv2.destroyAllWindows()
        return True
    elif key == 27: # `Esc` key
        pause_key = cv2.waitKey(0)
        if pause_key in [ord('d'), 27]:
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"{output_path} discarded")
            raise ExitException
        elif pause_key == ord('k'):
            cv2.destroyAllWindows()
            return True

def resize_and_crop_frame(frame, target_height, target_width, rotate_fit=False):
    try:
        # frame = image_utils.zoom_image_on_face(frame)
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

if __name__ == "__main__":
    main()

