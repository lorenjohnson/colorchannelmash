import cv2

class VideoReader:
    _instances = {}  # Class-level dictionary to store VideoReader instances

    def __init__(self, source_path):
        self.source_path = source_path
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            raise Exception(f"Error opening video file: {source_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, starting_frame):
        # Wraps around if starting frame position is beyond the length of the clip
        starting_frame %= self.total_frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error reading frame from video")
        return frame

    def release(self):
        self.cap.release()
        # Remove instance from the dictionary when released
        del VideoReader._instances[self.source_path]

    @classmethod
    def get_instance(cls, source_path):
        # Check if an instance already exists for the given source path
        if source_path not in cls._instances:
            # If not, create a new instance and store it in the dictionary
            cls._instances[source_path] = cls(source_path)
        return cls._instances[source_path]
