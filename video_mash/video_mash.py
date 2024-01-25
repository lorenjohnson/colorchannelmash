from typing import Dict
from pathlib import Path
import re
import os
import subprocess
import random
import argparse
import shlex
import json
import numpy as np
import cv2
from .video_source import VideoSource
from . import image_composition
from . import image_utils
from alive_progress import alive_bar

class ExitException(Exception):
    pass

class VideoMash:
    def __init__(
        self,
        source_paths,
        seconds,
        width,
        height,
        color_space,
        fps,
        webcam_enabled = False
    ):
        self.source_paths = source_paths
        self.seconds = seconds
        self.width = width
        self.height = height
        self.color_space = color_space
        self.fps = fps
        self.max_mash_number = 0
        self.target_brightness = 0.45
        self.target_contrast = 0.3
        self.selected_sources = []
        self.layer_mashes = [None]
        self.webcam_enabled = webcam_enabled

        # Check if there is at least one source path
        if source_paths:
            # Read metadata from the first path found in source_paths
            first_source_path = source_paths[0]
            metadata = self.read_metadata(first_source_path)
            # Check if the metadata contains the 'source_paths' key
            if metadata.get('tags').get('source_paths'):
                # Parse and set the values from metadata
                self.parse_metadata(metadata['tags'])
                
    # Extract values from metadata and set them in the VideoMash instance
    def parse_metadata(self, metadata):
        source_paths = metadata['source_paths'].split(',')
        starting_frames = list(map(int, metadata['starting_frames'].split(',')))

        self.source_paths = source_paths
        self.starting_frames = starting_frames
        self.seconds = float(metadata['seconds'])
        self.width = int(metadata['width'])
        self.height = int(metadata['height'])
        self.color_space = metadata['color_space']
        self.fps = float(metadata['fps'])

        for layer_index, source_path in enumerate(source_paths):
            self.selected_sources.append(
                VideoSource(source_path, starting_frames[layer_index])
            )
            processed_frame = self.get_and_process_frame(self.selected_sources[layer_index], layer_index)
            self.layer_mashes.append(
                self.mash_frames(self.layer_mashes[layer_index], processed_frame, layer_index)
            )

    def select_sources(self):
        selected_sources = self.selected_sources
        layer_mashes = self.layer_mashes
        layer_index = max(len(selected_sources) - 1, 0)
        preview_frame = None
        webcam_mode = False
        webcam_output_count = 0
        video_source = None

        while True:
            if len(selected_sources) >= layer_index + 1:
                video_source = selected_sources[layer_index]
            elif not video_source and not webcam_mode:
                selected_source_path = random.choice(self.source_paths)
                video_source = VideoSource(selected_source_path)
            elif not video_source and webcam_mode:
                video_source = self.capture_and_save_webcam()

            processed_frame = self.get_and_process_frame(video_source, layer_index)

            if processed_frame is None:
                video_source.release()
                video_source = None
                continue

            preview_frame = self.mash_frames(layer_mashes[layer_index], processed_frame, layer_index)
            cv2.imshow(f"Layer {layer_index}: (Space) Next Option | (Enter) Select | (Esc) Go back layer | (s) Start render | (c) Switch to Webcam", preview_frame)

            key = cv2.waitKey(0) & 0xFF

            # Enter - select current image as the layer and moves to next layer selection
            if key == 13:
                cv2.destroyAllWindows()
                layer_mashes.append(preview_frame.copy())
                selected_sources.append(video_source)
                video_source.release()
                video_source = None
                layer_index += 1
                continue
            # Esc - goes back a layer removing the previously selected source for that layer
            elif key == 27:
                if layer_index == 0:
                    video_source.release()
                    raise ExitException
                video_source.release()
                video_source = selected_sources.pop()
                layer_mashes.pop()
                layer_index -= 1
                continue
            # Space - shows next source option for this layer
            elif key == ord(' '):
                cv2.destroyAllWindows()
                if layer_index < len(selected_sources):
                    del selected_sources[layer_index]
                video_source.release()
                video_source = None
                continue
            # "c" - Switch to webcam mode
            elif key == ord('c'):
                cv2.destroyAllWindows()
                video_source.release()
                webcam_mode = not webcam_mode
                continue
            # "s" - Starts render
            elif key == ord('s') and layer_index > 0:
                cv2.destroyAllWindows()
                selected_sources.append(video_source)
                break

        return selected_sources

    def capture_and_save_webcam(self):
        # Set the video capture source to the default camera (0)
        cap = cv2.VideoCapture(0)

        # Set the frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Set the frame rate
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get the current time (in seconds)
        start_time = cv2.getTickCount() / cv2.getTickFrequency()

        frames = []

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Append the frame to the list
            frames.append(frame)

            # Show the webcam preview
            cv2.imshow("Webcam Preview - (q) to stop recording", frame)

            # Check if the specified duration has elapsed
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.seconds:
                break

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close the preview window
        cv2.destroyAllWindows()

        # Release the video capture object
        cap.release()

        # Convert the list of frames to a NumPy array
        webcam_video = np.array(frames)

        # Get the absolute path of the current working directory
        cwd = os.path.abspath(os.getcwd())

        # Determine the next available filename
        webcam_output_count = 1
        while os.path.exists(os.path.join(cwd, 'source', f'webcam-{webcam_output_count:03d}.mp4')):
            webcam_output_count += 1

        # Get the absolute path of the output filename
        output_filename = os.path.join(cwd, 'source', f'webcam-{webcam_output_count:03d}.mp4')

        # Save the webcam capture to the source directory
        writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

        # Create a VideoSource object for the webcam with an absolute path
        webcam_source = VideoSource(output_filename, 0)

        return webcam_source

    def mash(self, output_path):
        # Randomly select source videos and channel indices
        self.selected_sources = self.select_sources()
        try:
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          is_color = self.color_space.lower() not in ['gray']
          writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height), isColor=is_color)

          # Calculate the number of frames needed for the desired duration
          total_frames = int(self.fps * self.seconds)

          print("(Esc) Stop and delete video, keep generating sets")
          print("(k) Stop and keep video")

          with alive_bar(total_frames) as bar:
              for _ in range(total_frames):
                  mashed_frame = None

                  for index, video_source in enumerate(self.selected_sources):
                      processed_frame = self.get_and_process_frame(video_source, index)
                      if processed_frame is None:
                          continue

                      mashed_frame = self.mash_frames(mashed_frame, processed_frame, index)
                      video_source.starting_frame += 1

                  writer.write(mashed_frame)
                  bar()

                  if not mashed_frame is None:

                      preview_result = self.preview_frame(mashed_frame, output_path)
                      if preview_result == True:
                          break
                      elif preview_result == False:
                          return False

          writer.release()
          cv2.destroyAllWindows()
          self.write_metadata(output_path)

          return True

        finally:
          # Release VideoReader instances when done
          for video_source in self.selected_sources:
              video_source.release()

    def process_source_frame(self, frame, target_height, target_width, layer_index):
        # frame = image_utils.adjust_brightness(frame, self.target_brightness)
        frame = image_utils.adjust_contrast(frame, self.target_contrast)

        frame = self.resize_and_crop_frame(frame, target_height, target_width)
        # frame = image_utils.keep_color_channels_separated(frame)
        # frame = image_utils.apply_colormap(frame)
        return frame

    def mash_frames(self, provided_mashed_frame, new_frame, layer_index):
        channel_index = layer_index % 3
        is_color = self.color_space.lower() not in ['gray']

        if provided_mashed_frame is None:
            mashed_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            mashed_frame = provided_mashed_frame.copy()

        if is_color:
            # mashed_frame = image_composition.add_image_as_color_channel(mashed_frame, new_frame, channel_index)
            # mashed_frame = image_utils.glitch_frame(mashed_frame, new_frame)
            mashed_frame = image_composition.multiply([mashed_frame, new_frame])
            # Invert channels for HLS and YUV (usually a more useful result)
            # if self.color_space.lower() in ['hls', 'yuv']:
            #     mashed_frame[:, :, :] = 255 - mashed_frame[:, :, :]
            if self.color_space.lower() not in ['rgb', 'bgr']:
                mashed_frame = cv2.cvtColor(mashed_frame, getattr(cv2, f'COLOR_BGR2{self.color_space.upper()}'))
        else:
            # For grayscale, accumulate the intensity values
            mashed_frame[:, :, channel_index] += new_frame[:, :, channel_index]
            # Convert to grayscale color image
            mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_BGR2GRAY)
            mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_GRAY2BGR)
        # mashed_frame = image_utils.apply_colormap(mashed_frame, cv2.COLORMAP_HOT)

        return mashed_frame

    def resize_and_crop_frame(self, frame, target_height, target_width, rotate_fit=False):
        try:
            # frame = image_utils.zoom_image_on_face(frame)
            # Get frame dimensions
            height, width = frame.shape[:2]

            # # Check if a person is present
            # if image_utils.is_person_present(frame):
            #     # Adjust cropping region to move the center 25% up
            #     center_shift = int(0.8 * target_height)
            # else:
            center_shift = int(0.2 * target_height)

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
            start_y = max(0, (fill_height - target_height) // 2 - center_shift)
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
            print(f"Error in resize_and_crop_frame: {e}")
            return None

    def preview_frame(self, frame, output_path):
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

    def get_and_process_frame(self, video_source, layer_index):
        frame = video_source.get_frame()
        if frame is None:
            return None

        # Apply any processing to the frame
        frame = self.process_source_frame(frame, self.height, self.width, layer_index)
        return frame

    def write_metadata(self, output_path):
        output_path = Path(output_path)
        source_paths = [os.path.abspath(video_source.source_path) for video_source in self.selected_sources]
        starting_frames = [video_source.starting_frame for video_source in self.selected_sources]
        metadata = {
            "source_paths": ",".join(source_paths),
            "starting_frames": ",".join(map(str, starting_frames)),
            "seconds": self.seconds,
            "width": self.width,
            "height": self.height,
            "color_space": self.color_space,
            "fps": self.fps,
            "script_version": __version__ if '__version__' in globals() else "Unknown"
        }

        save_path = output_path.with_suffix('.metadata' + output_path.suffix)

        metadata_args = []
        for k, v in metadata.items():
            metadata_args.extend(['-metadata', f'{k}={v}'])

        args = [
            'ffmpeg',
            '-v', 'quiet',
            '-i', shlex.quote(str(output_path.absolute())),
            '-movflags',
            'use_metadata_tags',
            # '-map_metadata', '0',
            *metadata_args,
            '-c', 'copy',
            shlex.quote(str(save_path))
        ]

        try:
            # Run ffmpeg command
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Replace the original file with the new one
            os.replace(save_path, output_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running ffmpeg: {e.stderr.decode()}") from e
        finally:
            # Delete the save file if it still exists
            if os.path.exists(save_path):
                os.remove(save_path)

    def read_metadata(self, input_path: str) -> Dict:
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-print_format', 'json',
            '-show_format',
            shlex.quote(input_path)
        ]

        try:
            # Run ffprobe command and capture output
            result = subprocess.run(ffprobe_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_json = result.stdout.decode('utf-8')
            metadata_dict = json.loads(output_json)['format']
            return metadata_dict
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running ffprobe: {e.stderr.decode()}") from e
