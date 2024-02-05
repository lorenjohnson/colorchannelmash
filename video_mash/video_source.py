import os
import cv2
import random
import shutil

from alive_progress import alive_bar
from . import image_utils

class FileOpenException(Exception):
    pass

class GetFrameException(Exception):
    pass

class VideoSource:
    _instances = {}  # Class-level dictionary to store VideoSource instances

    def __init__(self, source_path, video_mash = None, starting_frame=None):
        self.source_path = source_path
        instance = self.create_or_get_instance(source_path, starting_frame)
        self.cap = instance.cap
        self.preprocessed_cap = None
        self.preprocessed_source_path = None
        self.current_frame = None
        self.total_frames = instance.total_frames
        self.starting_frame = instance.starting_frame
        self.video_mash = video_mash
        if not starting_frame:
            # Set starting_frame to a random valid point in the clip
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, self.total_frames - 1))

    @classmethod
    def create_or_get_instance(cls, source_path, starting_frame=None):
        # Check if an instance already exists for the given source path
        if source_path not in cls._instances:
            # If not, create a new instance and store it in the dictionary
            instance = cls._init_instance(source_path, starting_frame)
            cls._instances[source_path] = instance
        return cls._instances[source_path]

    @classmethod
    def _init_instance(cls, source_path, starting_frame=None):
        # Separate method to handle instance creation
        instance = cls.__new__(cls)
        instance.source_path = source_path
        instance.cap = cv2.VideoCapture(source_path)
        if not instance.cap.isOpened():
            raise FileOpenException(f"Error opening video file: {source_path}")
        instance.total_frames = int(instance.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        instance.starting_frame = starting_frame if starting_frame is not None else random.randint(0, instance.total_frames - 1)
        return instance

    def get_frame(self, starting_frame=None, preprocessing=False):
        cap = self.preprocessed_cap if self.preprocessed_cap else self.cap

        if starting_frame:
            starting_frame %= self.total_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        ret, frame = cap.read()

        if not ret and not preprocessing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        # TODO: Add error handling if necessary

        if not preprocessing and not self.preprocessed_cap:
            frame = self.process_frame(frame)

        self.current_frame = frame
        return frame

    def preprocess(self, layer_index, temp_dir):
        source_filename = os.path.basename(self.source_path)  # Get the filename of the source

        # Append "-resized" before the prefix and create a VideoWriter
        temp_file_path = os.path.join(temp_dir, f"{source_filename}-resized_temp_video_{layer_index}.mp4")
        
        # Initialize VideoCapture to read the video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_file_path, fourcc, self.video_mash.fps, (self.video_mash.width, self.video_mash.height), isColor=True)

        total_frames = int(self.video_mash.fps * self.video_mash.seconds)
        with alive_bar(total_frames) as bar:
            for _ in range(total_frames):
                # Read the next frame
                frame = self.get_frame(preprocessing=True)

                # Break the loop if no more frames are available
                if frame is None:
                    break

                processed_frame = self.process_frame(frame)

                # Write the processed frame to the temporary file
                writer.write(processed_frame)
                bar()

        self.cap.release()
        writer.release()
        
        shutil.copystat(self.source_path, temp_file_path)
        self.preprocessed_source_path = temp_file_path
        self.preprocessed_cap = cv2.VideoCapture(self.preprocessed_source_path)

        return True

    def process_frame(self, frame):
        # if self.brightness:
        #     frame = image_utils.adjust_brightness(frame, self.brightness)
        # if self.contrast:
        #     frame = image_utils.adjust_contrast(frame, self.contrast)
        frame = image_utils.resize_and_crop(frame, self.video_mash.height, self.video_mash.width)
        # frame = image_utils.keep_color_channels_separated(frame)
        # frame = image_utils.apply_colormap(frame)
        return frame

    def release(self):
        if self.cap: self.cap.release()
        # Release and delete preprocessed
        if self.preprocessed_cap:
            self.preprocessed_cap.release()
            try:
                # Attempt to remove the directory
                shutil.rmtree(os.path.dirname(self.preprocessed_source_path))
            except FileNotFoundError:
                # Ignore the error if the directory was already deleted
                pass        # Remove instance from the dictionary if exists
        if self.source_path in self._instances:
            del self._instances[self.source_path]
