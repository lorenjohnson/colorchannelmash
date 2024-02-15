import random
import tempfile

import cv2
import numpy as np
from alive_progress import alive_bar

from . import blend_modes
from . import effects
from . import colormaps
from . import video_mash_metadata
from .video_source import VideoSource, FileOpenException
from .webcam_capture import WebcamCapture

class ExitException(Exception):
    pass

class VideoMash:
    def __init__(self, **kwargs):
        default_values = {
            'source_paths': None,
            'output_path': None,
            'seconds': 10,
            'width': 1242,
            'height': 2688,
            'opacity': 1,
            'fps': 12,
            'mode': 'multiply',
            # TODO: Implement effects, currently only color_mode
            'effects': ['rgb'],
            'colormap': None,
            # TODO: Currently ignored
            'brightness': 0.45,
            'contrast': 0.2,
            'webcam_enabled': True,
            # Not yet implemented to CLI
            'auto': 1,
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
                self.colormap = mash_data.get('colormap')
                for index, source_path in enumerate(mash_data.get('source_paths')):
                    starting_frame = mash_data.get('starting_frames')[index]
                    self.selected_sources.append(
                        VideoSource(source_path, video_mash=self, starting_frame=starting_frame)
                    )

        if self.auto > 1:
            self.random_sources(self.auto)

    def random_sources(self, layers):
        for _ in range(layers):
            random_source_path = random.choice(self.source_paths)
            current_source = VideoSource(random_source_path, self)
            self.selected_sources.append(current_source)
            self.effects = random.choice(effects.EFFECT_COMBOS)
            self.mode = random.choice(blend_modes.BLEND_MODES)
            self.colormap = random.choice(list(colormaps.COLORMAPS.keys()))
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
            cv2.namedWindow(window_title)
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_title, current_mash)

            # cv2.resizeWindow(window_title, 1280, 720)
            key = cv2.waitKey(0) & 0xFF

            # Enter - select current image as the layer and moves to next layer selection
            if key == 13:
                layer_index += 1
                continue
            # Esc - goes back a layer removing the previously selected source for that layer
            elif key == 27:
                source = self.selected_sources.pop()
                source.release()
                if layer_index == 0:
                    raise ExitException
                layer_index -= 1
                cv2.destroyAllWindows()
                continue
            # Space - shows next source option for this layer
            elif key == ord(' '):
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
            elif key == ord('f'):
                cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('d'):
                self.reset()
                self.random_sources(self.auto)
                continue
            # "p" - Save current frame as a PNG
            elif key == ord('p'):
                self.save_screenshot(current_mash)
            # "s" - Starts render
            elif key == ord('s'):
                cv2.destroyAllWindows()
                break
            # "l" - Cycle LUTs / Colormaps (all layers)
            elif key == ord('l'):
                self.cycle_colormap()
            # "m" - Cycle Blend Modes (all layers)
            elif key == ord('m'):
                self.cycle_blend_mode()
            elif key == ord('M'):
                self.cycle_blend_mode(True)
            # ',' Cycle blend opacity (all layers)
            elif key == ord(','):
                self.opacity = (self.opacity + 0.1) % 1.0
            # "e" - Next Effect
            elif key == ord('e'):
                self.cycle_effect()
            elif ord('0') <= key <= ord('8'):
                self.apply_effect(key)
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
                frame = source.get_current_frame_or_next()
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

    def cycle_colormap(self, backward=False):
        if self.colormap is None:
            next_colormap = list(colormaps.COLORMAPS.keys())[0]
        else:
            current_index = list(colormaps.COLORMAPS.keys()).index(self.colormap)
            direction = -1 if backward else 1
            colormap_keys = list(colormaps.COLORMAPS.keys())
            next_index = (current_index + direction) % len(colormap_keys)
            next_colormap = colormap_keys[next_index]
        self.colormap = next_colormap
        print(f"LUT: {self.colormap}")

    def cycle_effect(self, backward=False):
        effects_count = len(self.effects)
        current_index = effects.EFFECT_COMBOS.index(self.effects) if effects_count > 0 and self.effects in effects.EFFECT_COMBOS else 0
        direction = -1 if backward else 1
        next_index = (current_index + direction) % len(effects.EFFECT_COMBOS)
        self.effects = effects.EFFECT_COMBOS[next_index]
        print(f"Effects: {self.effects}")

    def cycle_blend_mode(self, backward=False):
        current_index = blend_modes.BLEND_MODES.index(self.mode)
        direction = -1 if backward else 1
        next_index = (current_index + direction) % len(blend_modes.BLEND_MODES)
        self.mode = blend_modes.BLEND_MODES[next_index]
        print(f"Mode: {self.mode}")
    
    def apply_effect(self, key):
        if key == ord('0'):
            self.effects = []
        else:
            effect_index = key - ord('1') # Adjust the index to match the list
            # self.effects.append(effects.EFFECTS[effect_index])      
            if effects.EFFECTS[effect_index] in self.effects:
                self.effects.remove(effects.EFFECTS[effect_index])
            else:
                self.effects.append(effects.EFFECTS[effect_index])
        print(f"Effects: {self.effects}")
    
    def mash(self):
        try:
            # Randomly select source videos and channel indices
            # if len(self.selected_sources) < 1:
            self.select_layers()

            if self.preprocess: self.preprocess_selected_sources()
            for selected_source in self.selected_sources:
                print(selected_source.source_path)
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

                    for index, source in enumerate(self.selected_sources):
                        current_frame = source.get_frame()

                        if current_frame is None:
                            continue

                        mashed_frame = self.mash_frames(mashed_frame, current_frame, index)

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
            self.reset()

    def preview_frame(self, frame):
        window_title = "(Esc) Pause | (d) Stop and Delete | (k) Stop and Keep"
        cv2.namedWindow(window_title)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_title, frame)
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
        mashed_frame = blend_modes.apply(self.mode, provided_mashed_frame, new_frame, layer_index, self.opacity)

        for effect in self.effects:
            mashed_frame = effects.apply(effect, mashed_frame)

        if (self.colormap):
            mashed_frame = colormaps.apply(mashed_frame, self.colormap)

        return mashed_frame

    def reset(self):
        # Release original sources
        for selected_source in self.selected_sources:
            selected_source.release()
        self.selected_sources = []

    # Source Preprocessing

    def preprocess_selected_sources(self):
        # Create a temporary directory to store processed video files
        temp_dir = tempfile.mkdtemp(prefix="VideoMash-")

        for layer_index, selected_source in enumerate(self.selected_sources):
            selected_source.preprocess(layer_index, temp_dir)

        return True
