import cv2
import random

class FileOpenException(Exception):
    pass

class GetFrameException(Exception):
    pass

class VideoSource:
    _instances = {}  # Class-level dictionary to store VideoSource instances

    def __init__(self, source_path, starting_frame=None):
        self.source_path = source_path
        instance = self.create_or_get_instance(source_path, starting_frame)
        self.cap = instance.cap
        self.total_frames = instance.total_frames
        self.starting_frame = instance.starting_frame  # Add this line

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

    def get_frame(self, starting_frame=None):
        if starting_frame:
            starting_frame %= self.total_frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        ret, frame = self.cap.read()

        # TODO: Add error handling if necessary

        return frame

    def release(self):
        self.cap.release()
        # Remove instance from the dictionary if exists
        if self.source_path in self._instances:
            del self._instances[self.source_path]
