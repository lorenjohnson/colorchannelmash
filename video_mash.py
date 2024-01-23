from __init__ import __version__
from typing import Dict
from pathlib import Path
import os
import subprocess
import random
import argparse
import shlex
import json
import numpy as np
import cv2
from video_source import VideoSource
import frame_effects
import image_composition
import image_utils
from alive_progress import alive_bar

class ExitException(Exception):
    pass

class VideoMash:
    def __init__(self, source_paths, seconds, width, height, color_space, fps):
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

        for index, source_path in enumerate(source_paths):
            print(f"source_path {source_path} | starting_frames[index] {starting_frames[index]}")
            self.selected_sources.append(
                VideoSource(source_path, starting_frames[index])
            )

    def select_sources(self):
        selected_sources = self.selected_sources
        layer_index = len(selected_sources) - 1
        layer_mashes = [None]
        preview_frame = None
        if layer_index > 0:
            video_source = selected_sources[layer_index]
        else:
            video_source = None

        while True:
            if not video_source:
                selected_source_path = random.choice(self.source_paths)
                video_source = VideoSource(selected_source_path)

            processed_frame = self.get_and_process_frame(video_source, layer_index)

            if processed_frame is None:
                video_source.release()
                video_source = None
                continue

            preview_frame = self.mash_frames(layer_mashes[layer_index], processed_frame, layer_index)
            cv2.imshow(f"Layer {layer_index + 1}: (Space) Next Option | (Enter) Select | (Esc) Go back layer | (s) Start render", preview_frame)

            key = cv2.waitKeyEx(0) & 0xFF
            print(key)

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
                if (layer_index == 0):
                    video_source.release()
                    raise ExitException
                video_source.release()
                video_source = selected_sources.pop()
                layer_mashes.pop()
                layer_index -= 1
            # Space - shows next source option for this layer
            elif key == ord(' '):
                cv2.destroyAllWindows()
                video_source.release()
                video_source = None
            # "s" - Starts render
            elif key == ord('s') and layer_index > 0:
                cv2.destroyAllWindows()
                selected_sources.append(video_source)
                break

        return selected_sources

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
        # frame = frame_effects.keep_them_separated_alt(frame)
        # frame = frame_effects.apply_colormap(frame, cv2.COLORMAP_HOT)
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
        # mashed_frame = frame_effects.apply_colormap(mashed_frame, cv2.COLORMAP_HOT)

        return mashed_frame

    def resize_and_crop_frame(self, frame, target_height, target_width, rotate_fit=False):
        try:
            # frame = frame_effects.zoom_frame_on_face(frame)
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
