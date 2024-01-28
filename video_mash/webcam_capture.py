from typing import List
import cv2
import os
import shlex
import subprocess
import numpy as np

class WebcamCapture:
    def __init__(self, width, height, fps, seconds):
        self.width = width
        self.height = height
        self.fps = fps
        self.seconds = seconds
        
        cwd = os.path.abspath(os.getcwd())
        webcam_output_count = 1

        while os.path.exists(os.path.join(cwd, 'source', f'webcam-{webcam_output_count:03d}.mp4')):
            webcam_output_count += 1

        self.output_filename = os.path.join(cwd, 'source', f'webcam-{webcam_output_count:03d}.mp4')

    def capture_and_save_webcam(self):
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        start_time = cv2.getTickCount() / cv2.getTickFrequency()
        frames = []

        while True:
            ret, frame = cap.read()
            frames.append(frame)
            cv2.imshow("Webcam Preview - (q) to stop recording", frame)

            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.seconds:
                break

            if cv2.waitKey(1) & 0xFF == 13:
                break

        cv2.destroyAllWindows()
        cap.release()

        webcam_video = np.array(frames)
        self.save_webcam_capture(webcam_video)

        return self.output_filename

    def save_webcam_capture(self, frames: List[np.ndarray]):
        writer = cv2.VideoWriter(str(self.output_filename), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        for frame in frames:
            writer.write(frame)
        writer.release()
