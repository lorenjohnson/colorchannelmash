from video_reader import VideoReader

class VideoSource:
    def __init__(self, source_path, starting_frame):
        self.source_path = source_path
        self.starting_frame = starting_frame
        self.video_reader = VideoReader.get_instance(source_path)

    def get_frame(self):
        return self.video_reader.get_frame(self.starting_frame)

    def release(self):
        pass  # VideoReader instances will be managed centrally
