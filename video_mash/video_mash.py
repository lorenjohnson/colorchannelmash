import random
import tempfile

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

    EFFECTS = ['rgb', 'hsv', 'hls', 'yuv', 'gray', 'invert', 'jpeg', 'ocean']
    EFFECT_COMBOS = [
        [],
        ['hls', 'rgb'],
        ['invert', 'hls', 'rgb'],
        ['hsv', 'rgb'],
        ['invert', 'hsv', 'rgb'],
        ['yuv', 'rgb'],
        ['invert', 'yuv', 'rgb'],
    ] + [[effect] for effect in EFFECTS]
    print(EFFECT_COMBOS)
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
            'auto': False,
            'preprocess': False
        }

        # Create a new dictionary with default values
        init_values = default_values.copy()

        # Update init_values with provided keyword arguments
        init_values.update(kwargs)
        
        if not len(init_values['source_paths']) > 0:
            raise Exception("At least one source_path must be provided")

        if not init_values['output_path']:
            raise Exception("Must provide an output_path")

        # Initialize attributes with init_values
        self.__dict__.update(init_values)

        # Additional initialization
        if self.webcam_enabled:
            self.webcam_capture = WebcamCapture(self.width, self.height, self.fps, self.seconds)
        self.selected_sources = []

        if len(self.source_paths) == 1:
            mash_data = video_mash_metadata.read(self.source_paths[0])
            if mash_data:
                self.seconds = mash_data.get('seconds')
                self.effects = mash_data.get('effects')
                self.mode = mash_data.get('mode')
                self.fps = mash_data.get('fps')
                for index, source_path in enumerate(mash_data.get('source_paths')):
                    starting_frame = mash_data.get('starting_frames')[index]
                    self.selected_sources.append(
                        VideoSource(source_path, video_mash=self, starting_frame=starting_frame)
                    )

        # if self.auto:
        #     self.random_sources()

    def random_sources(self):
        for _ in range(3):
            random_source_path = random.choice(self.source_paths)
            current_source = VideoSource(random_source_path, self)
            self.selected_sources.append(current_source)
        return self.selected_sources


    def select_layers(self):
        layer_index = max(len(self.selected_sources) - 1, 0)

        while True:
            try:
                if layer_index >= len(self.selected_sources):
                    random_source_path = random.choice(self.source_paths)
                    random_source = VideoSource(random_source_path, self)
                    self.selected_sources.append(random_source)
            except FileOpenException as e:
                print(f"Can't open source {random_source_path}, skipping")
                continue

            current_mash = self.preview_layers(layer_index)

            window_title = f"Layer {layer_index}: (Space) Next Option | (Enter) Select | (Esc) Go back layer | (s) Start render | (c) Switch to Webcam"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.imshow(window_title, current_mash)
            cv2.resizeWindow(window_title, 1280, 720)
            key = cv2.waitKey(0) & 0xFF

            # Enter - select current image as the layer and moves to next layer selection
            if key == 13:
                cv2.destroyAllWindows()
                layer_index += 1
                continue
            # Esc - goes back a layer removing the previously selected source for that layer
            elif key == 27:
                cv2.destroyAllWindows()
                source = self.selected_sources.pop()
                source.release()
                if layer_index == 0:
                    raise ExitException
                layer_index -= 1
                continue
            # Space - shows next source option for this layer
            elif key == ord(' '):
                cv2.destroyAllWindows()
                self.selected_sources[layer_index].release()
                del self.selected_sources[layer_index]
                continue
            # "c" - Source from webcam
            elif key == ord('c'):
                print("beginning capture from webcam")
                cv2.destroyAllWindows()
                self.selected_sources[layer_index].release()
                webcam_capture_output = self.webcam_capture.capture_and_save_webcam()
                self.selected_sources[layer_index] = VideoSource(webcam_capture_output, video_mash=self, starting_frame=0)
                continue
            # "p" - Save current frame as a PNG
            elif key == ord('p'):
                self.save_screenshot(current_mash)
            # "s" - Starts render
            elif key == ord('s'):
                cv2.destroyAllWindows()
                break
            # "m" - Next Mode
            elif key == ord('m'):
                self.get_next_mode()
            else:
                # "e" - Next Effect
                if key == ord('e'):
                    self.get_next_effect()
                elif key == ord('0'):
                    self.effects = []
                    print(self.effects)
                elif ord('1') <= key <= ord('8'):
                    effect_index = key - ord('1') # Adjust the index to match the list
                    if self.EFFECTS[effect_index] in self.effects:
                        self.effects.remove(self.EFFECTS[effect_index])
                    else:
                        self.effects.append(self.EFFECTS[effect_index])
                    print(self.effects)
                else:
                    print(key)

        return self.selected_sources

    def preview_layers(self, layer_index = None):
        layer_index = len(self.selected_source) if layer_index is None else layer_index
        mashed_frame = None

        for index in range(layer_index + 1):
            previous_mash = mashed_frame

            if index < len(self.selected_sources):
                source = self.selected_sources[index]
                frame = source.get_frame() if source.current_frame is None else source.current_frame
                frame = source.process_frame(frame)
                mashed_frame = self.mash_frames(previous_mash, frame, index)

        return mashed_frame

    def save_screenshot(self, mashed_frame):
        output_filename = self.output_path.stem  # Get the filename without extension

        existing_screenshots = [file for file in self.output_path.parent.glob(f"{output_filename}.screenshot-*.png")]

        screenshot_number = 1
        if existing_screenshots:
            latest_screenshot = max(existing_screenshots)
            parts = latest_screenshot.stem.split('-')
            screenshot_number = int(parts[-1]) + 1

        screenshot_filename = f"{output_filename}.screenshot-{screenshot_number:03d}.png"
        screenshot_path = self.output_path.parent / screenshot_filename
        cv2.imwrite(str(screenshot_path), mashed_frame)
        print(f"Frame saved as {screenshot_path}")


    def get_next_effect(self):
        effects_count = len(self.effects)
        current_index = self.EFFECT_COMBOS.index(self.effects) if effects_count > 0 and self.effects in self.EFFECT_COMBOS else 0
        next_index = (current_index + 1) % len(self.EFFECT_COMBOS)
        self.effects = self.EFFECT_COMBOS[next_index]
        print(f"Effects: {self.effects}")

        return self.effects

    def get_next_mode(self):
        current_index = self.MODES.index(self.mode)
        next_index = (current_index + 1) % len(self.MODES)
        self.mode = self.MODES[next_index]
        print(f"Mode: {self.mode}")
        return self.mode

    def mash(self):
        try:
            # Randomly select source videos and channel indices
            # if len(self.selected_sources) < 1:
            self.select_layers()

            if self.preprocess: self.preprocess_selected_sources()

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

                    for index, video_source in enumerate(self.selected_sources):
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
        cv2.namedWindow(rendering_title, cv2.WINDOW_NORMAL)
        cv2.imshow(rendering_title, frame)
        cv2.resizeWindow(rendering_title, 1280, 720)
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
                self.select_layers()
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

            # NOTE: Not necessary to convert new_frame back at this time
            # new_frame = new_frame[:, :, :3]
            # new_frame = new_frame.astype(np.uint8)

        # TODO: Add as an effect option
        for effect in self.effects:
            if effect.lower() in ['jpeg']:
                mashed_frame = image_utils.mjpeg_compression(mashed_frame, 45)
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

    # Source Preprocessing

    def preprocess_selected_sources(self):
        # Create a temporary directory to store processed video files
        temp_dir = tempfile.mkdtemp(prefix="VideoMash-")

        for layer_index, selected_source in enumerate(self.selected_sources):
            selected_source.preprocess(layer_index, temp_dir)

        return True
