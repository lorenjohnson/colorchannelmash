import os
import random
import tempfile
import shutil

import cv2
import numpy as np
from alive_progress import alive_bar
import blend_modes

from . import blend_modes as mash_blend_modes
from . import image_utils
from . import video_mash_metadata
from .video_source import VideoSource, FileOpenException
from .webcam_capture import WebcamCapture

class ExitException(Exception):
    pass

class VideoMash:
    MODES = ['channels', 'accumulate', 'soft_light', 'lighten_only', 'dodge', 'addition', 'darken_only', 'multiply', 'hard_light',
            'difference', 'subtract', 'grain_extract', 'grain_merge', 'divide', 'overlay', 'normal']

    EFFECTS = ['hsv', 'hls', 'yuv', 'gray', 'invert', 'ocean', 'rgb']

    def __init__(self, **kwargs):
        default_values = {
            'source_paths': None,
            'output_path': None,
            'seconds': 10,
            'width': 1242,
            'height': 2688,
            'opacity': 0.5,
            'fps': 12,
            'mode': 'multiply',
            # TODO: Implement effects, currently only color_mode
            'effects': ['rgb'],
            # TODO: Currently ignored
            'brightness': 0.45,
            'contrast': 0.2,
            'webcam_enabled': True,
            # Not yet implemented to CLI
            'auto': False
        }

        # Create a new dictionary with default values
        init_values = default_values.copy()

        # Update init_values with provided keyword arguments
        init_values.update(kwargs)
        
        # TODO: Test and Fix mashfile handling
        # TODO: Add JSON mashfile handling
        # Update from mashfile if exists, set remash to True
        # remash = False
        # if len(source_paths) == 1:
        #     mash_data = video_mash_metadata.read(mash_file)
        #     if mash_data:
        #         remash = True
        #         init_values.update(mash_data)

        if not len(init_values['source_paths']) > 0:
            raise Exception("At least one source_file must be provided")

        if not init_values['output_path']:
            raise Exception("Must provide an output_path")

        # Initialize attributes with init_values
        self.__dict__.update(init_values)

        # Additional initialization
        if self.webcam_enabled:
            self.webcam_capture = WebcamCapture(self.width, self.height, self.fps, self.seconds)
        self.selected_sources = []
        self.preprocessed_selected_sources = []
        self.layer_mashes = []

        # TODO: Fix mash_file handling
        # if remash:
        #     for layer_index, source_path in enumerate(self.source_paths):
        #         source = VideoSource(source_path, self.starting_frames[layer_index])
        #         self.selected_sources.append(source)
        #         current_frame = self.get_source_frame(source, layer_index)
        #         self.layer_mashes.append(
        #             self.mash_frames(self.layer_mashes[layer_index], current_frame, layer_index)
        #         )
        if self.auto:
            self.random_sources()
            

    def random_sources(self):
        for _ in range(3):
            random_source_path = random.choice(self.source_paths)
            current_source = VideoSource(random_source_path)
            self.selected_sources.append(current_source)
        return self.selected_sources

    def select_sources(self):
        layer_index = max(len(self.selected_sources) - 1, 0)
        current_source = None

        while True:
            try:
                if not current_source:
                    random_source_path = random.choice(self.source_paths)
                    current_source = VideoSource(random_source_path)
            except FileOpenException as e:
                print(f"Can't open source {random_source_path}, skipping")
                continue

            try:
                if layer_index > 0:
                    previous_mash = self.layer_mashes[layer_index - 1] 
                else:
                    previous_mash = None

                current_frame = current_source.get_frame()
                current_frame = self.process_source_frame(current_frame)

                next_mash = self.mash_frames(previous_mash, current_frame, layer_index)
                if next_mash.shape[0] < 10:
                    raise Exception("Preview frame is empty (0 height)")
            except Exception as e:
                print(f"Can't get preview frame: {e}")
                current_source.release()
                current_source = None
                continue

            window_title = f"Layer {layer_index}: (Space) Next Option | (Enter) Select | (Esc) Go back layer | (s) Start render | (c) Switch to Webcam"
            cv2.imshow(window_title, next_mash)
            key = cv2.waitKey(0) & 0xFF

            # Enter - select current image as the layer and moves to next layer selection
            if key == 13:
                cv2.destroyAllWindows()
                self.layer_mashes.append(next_mash.copy())
                self.selected_sources.append(current_source)
                current_source = None
                layer_index += 1
                continue
            # Esc - goes back a layer removing the previously selected source for that layer
            elif key == 27:
                if layer_index == 0:
                    current_source.release()
                    raise ExitException
                current_source.release()
                current_source = self.selected_sources.pop()
                self.layer_mashes.pop()
                layer_index -= 1
                continue
            # Space - shows next source option for this layer
            elif key == ord(' '):
                cv2.destroyAllWindows()
                if layer_index < len(self.selected_sources):
                    del self.selected_sources[layer_index]
                current_source.release()
                current_source = None
                continue
            # "c" - Source from webcam
            elif key == ord('c'):
                print("beginning capture from webcam")
                cv2.destroyAllWindows()
                current_source.release()
                webcam_capture_output = self.webcam_capture.capture_and_save_webcam()
                current_source = VideoSource(webcam_capture_output, 0)
                continue
            # "s" - Starts render
            elif key == ord('s') and layer_index > 0:
                cv2.destroyAllWindows()
                self.selected_sources.append(current_source)
                break
            elif key == ord('e'):
                # layer_index = 0
                self.get_next_effect()
            elif key == ord('m'):
                # layer_index = 0
                self.get_next_mode()

        return self.selected_sources

    def get_next_effect(self):
        effects_count = len(self.effects)
        if effects_count > 0:
            curent_index = self.EFFECTS.index(self.effects[effects_count - 1])
            next_index = (curent_index + 1) % len(self.EFFECTS)
            self.effects[effects_count - 1] = self.EFFECTS[next_index]
        else:
            self.effects[self.EFFECTS[0]]
            
        return self.effects

    def get_next_mode(self):
        current_index = self.MODES.index(self.mode)
        next_index = (current_index + 1) % len(self.MODES)
        self.mode = self.MODES[next_index]
        return self.mode

    def mash(self):
        try:
            # Randomly select source videos and channel indices
            if len(self.selected_sources) < 1:
                self.select_sources()

            self.preprocess_selected_sources()

            # try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height), isColor=True)
        
            # Calculate the number of frames needed for the desired duration
            total_frames = int(self.fps * self.seconds)

            print("(Esc) Stop and delete video, keep generating sets")
            print("(k) Stop and keep video")

            with alive_bar(total_frames) as bar:
                for _ in range(total_frames):
                    mashed_frame = None

                    for index, video_source in enumerate(self.preprocessed_selected_sources):
                        current_frame = video_source.get_frame()

                        if current_frame is None:
                            continue

                        mashed_frame = self.mash_frames(mashed_frame, current_frame, index)
                        video_source.starting_frame += 1

                    writer.write(mashed_frame)
                    bar()

                    if not mashed_frame is None:
                        preview_result = self.preview_frame(mashed_frame)
                        if preview_result == True:
                            break
                        elif preview_result == False:
                            return False

            writer.release()
            cv2.destroyAllWindows()
            video_mash_metadata.write(self)

            return True

        finally:
            self.cleanup()

    def preview_frame(self, frame):
        rendering_title = "(Esc) Pause | (d) Stop and Delete | (k) Stop and Keep"
        cv2.imshow(rendering_title, frame)
        key = cv2.waitKey(1)

        if key ==  ord('d'):
            print("Generation stopped", end="")
            if self.output_path.exists():
                self.output_path.unlink()
                print(f", {self.output_path} discarded")
            cv2.destroyAllWindows()
            return False
        elif key == ord('k'):
            print(f"Generation stopped, {self.output_path} saved.")
            cv2.destroyAllWindows()
            return True
        elif key == 27: # `Esc` key
            pause_key = cv2.waitKey(0)
            if pause_key == 27:
                if self.output_path.exists():
                    self.output_path.unlink()
                    print(f"{self.output_path} discarded, recomposing from here")
                cv2.destroyAllWindows()
                self.select_sources()
                # return False
            elif pause_key == ord('q'):
                if self.output_path.exists():
                    self.output_path.unlink()
                    print(f"{self.output_path} discarded")
                raise ExitException
            # elif pause_key in [ord('d'), 27]:
            # elif pause_key == ord('k'):
            #     cv2.destroyAllWindows()
            #     return True

    def mash_frames(self, provided_mashed_frame, new_frame, layer_index):
        channel_index = layer_index % 3

        # setup for blend_modes
        mode = self.mode

        if provided_mashed_frame is None:
            if mode in ['channels']:
                # Black Image
                # For blend modes that rely on the content of the first image to produce meaningful results, setting the first
                # image to all black may result in the second image dominating the blend. Blend modes that involve multiplication
                # or darkening effects may be suitable for an all-black first image.
                mashed_frame = np.zeros_like(new_frame)
            elif mode in ['add', 'lighten_only']:
                # All White Image
                # Blend modes that involve addition or lightening effects might be suitable for an all-white first image.
                # Setting the first image to all white can be a good choice when you want the second image to have a strong influence
                # on the result.
                mashed_frame = np.ones_like(new_frame) * 255
            else:
                # All Gray Image
                # An all-gray first image (mid-gray, RGB(128, 128, 128)) can be a neutral starting point. It may not bias the blend toward
                # dark or light, and it can be used as a baseline for various blend modes. This can be suitable for blend modes that involve
                # both lightening and darkening effects, such as overlay or soft light.
                mashed_frame = np.ones_like(new_frame) * 128
        else:
            mashed_frame = provided_mashed_frame.copy()

        if mode in ['channels']:
            mashed_frame = mash_blend_modes.add_image_as_color_channel(mashed_frame, new_frame, channel_index)
        # Previously only used for "gray": accumulate the intensity values
        elif mode in ['accumulate']:
            mashed_frame[:, :, channel_index] += new_frame[:, :, channel_index]
        else:
            mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_BGR2BGRA)
            mashed_frame = mashed_frame.astype(float)

            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2BGRA)
            new_frame = new_frame.astype(float)

            blend_mode = getattr(blend_modes, mode)
            mashed_frame = blend_mode(mashed_frame, new_frame, self.opacity)
            
            mashed_frame = mashed_frame[:, :, :3]
            mashed_frame = mashed_frame.astype(np.uint8)

            new_frame = new_frame[:, :, :3]
            new_frame = new_frame.astype(np.uint8)

        for effect in self.effects:
            if effect.lower() in ['invert']:
                mashed_frame[:, :, :] = 255 - mashed_frame[:, :, :]
            if effect.lower() in ['hls', 'yuv']:
                mashed_frame = cv2.cvtColor(mashed_frame, getattr(cv2, f'COLOR_BGR2{effect.upper()}'))
            if effect.lower() in ['gray']:
                mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_BGR2GRAY)
                mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_GRAY2BGR)
            if effect.lower() in ['ocean']:
                mashed_frame = image_utils.apply_colormap(mashed_frame, cv2.COLORMAP_OCEAN)
            if effect.lower() in ['rgb']:
                mashed_frame = cv2.cvtColor(mashed_frame, cv2.COLOR_BGR2RGB)
        return mashed_frame

    def cleanup(self):
        # Release original sources
        for selected_source in self.selected_sources:
            selected_source.release()
        # Release and delete preprocessed selected sources
        if len(self.preprocessed_selected_sources) > 0:
            shutil.rmtree(os.path.dirname(self.preprocessed_selected_sources[0].source_path))
            for selected_source_preprocessed in self.preprocessed_selected_sources:
                selected_source_preprocessed.release()

    # Source Preprocessing

    def process_source_frame(self, frame):
        # if self.brightness:
        #     frame = image_utils.adjust_brightness(frame, self.brightness)
        # if self.contrast:
        #     frame = image_utils.adjust_contrast(frame, self.contrast)

        frame = self.resize_and_crop_source_frame(frame)
        # frame = image_utils.keep_color_channels_separated(frame)
        # frame = image_utils.apply_colormap(frame)
        return frame

    def resize_and_crop_source_frame(self, frame, rotate_fit=False):
        try:
            target_height = self.height
            target_width = self.width

            # frame = image_utils.zoom_image_on_face(frame)
            # Get frame dimensions
            height, width = frame.shape[:2]

            # # Check if a person is present
            # if image_utils.is_person_present(frame):
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

            # Resize the frame maintaining aspect ratio
            resized_frame = cv2.resize(frame, (fill_width, fill_height))

            # Check if rotating the image would result in more coverage
            if rotate_fit and fill_height < target_height:
                fill_width, fill_height = fill_height, fill_width

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
            print(f"Error in resize_and_crop_source_frame: {e}")
            return None

    def preprocess_source(self, source, layer_index, temp_dir):
        source_path = source.source_path
        source_filename = os.path.basename(source_path)  # Get the filename of the source

        # Append "-resized" before the prefix and create a VideoWriter
        temp_file_path = os.path.join(temp_dir, f"{source_filename}-resized_temp_video_{layer_index}.mp4")
        
        # Initialize VideoCapture to read the video file
        # video_capture = cv2.VideoCapture(source_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_file_path, fourcc, 30, (self.width, self.height), isColor=True)

        total_frames = int(self.fps * self.seconds)
        with alive_bar(total_frames) as bar:
            for _ in range(total_frames):
                # Read the next frame
                frame = source.get_frame()

                # Break the loop if no more frames are available
                if frame is None:
                    break

                processed_frame = self.process_source_frame(frame)

                # Write the processed frame to the temporary file
                writer.write(processed_frame)
                bar()

        # Release the VideoCapture and VideoWriter instances
        source.release()
        writer.release()

        shutil.copystat(source_path, temp_file_path)

        return VideoSource(temp_file_path, source.starting_frame)

    def preprocess_selected_sources(self):
        # Create a temporary directory to store processed video files
        temp_dir = tempfile.mkdtemp(prefix="VideoMash-")

        for layer_index, selected_source in enumerate(self.selected_sources):
            self.preprocessed_selected_sources.append(
                self.preprocess_source(selected_source, layer_index, temp_dir)
            )

        return True
