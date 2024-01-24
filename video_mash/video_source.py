import random
import cv2
from .video_reader import VideoReader

class VideoSource:
    def __init__(self, source_path, starting_frame = None):
        self.source_path = source_path
        self.video_reader = VideoReader.get_instance(source_path)
        if self.video_reader.cap:
            if starting_frame:        
                self.starting_frame = starting_frame
            else:
                self.starting_frame = random.randint(0, int(self.video_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

    def get_frame(self):
        return self.video_reader.get_frame(self.starting_frame)

    def release(self):
        pass  # VideoReader instances will be managed centrally
