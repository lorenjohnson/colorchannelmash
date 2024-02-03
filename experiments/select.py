import tkinter as tk
from tkinter import filedialog
import os
import json

class VideoMetadataCreator:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Metadata Creator")

        # Button to open file dialog for selecting videos
        self.video_button = tk.Button(master, text="Select Videos", command=self.select_videos)
        self.video_button.pack()

        # Button to open file dialog for selecting photos
        self.photo_button = tk.Button(master, text="Select Photos", command=self.select_photos)
        self.photo_button.pack()

        # Button to create metadata file
        self.create_button = tk.Button(master, text="Create Metadata", command=self.create_metadata)
        self.create_button.pack()

        # Lists to store selected videos and photos
        self.selected_videos = []
        self.selected_photos = []

    def select_videos(self):
        videos = filedialog.askopenfilenames(title="Select Videos", filetypes=[("Video files", "*.mp4;*.avi")])
        self.selected_videos = list(videos)

    def select_photos(self):
        photos = filedialog.askopenfilenames(title="Select Photos", filetypes=[("Image files", "*.jpg;*.png")])
        self.selected_photos = list(photos)

    def create_metadata(self):
        metadata = {
            "videos": self.selected_videos,
            "photos": self.selected_photos
        }

        # Save metadata to a JSON file
        metadata_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if metadata_file:
            with open(metadata_file, 'w') as json_file:
                json.dump(metadata, json_file)

            print(f"Metadata file saved to: {metadata_file}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoMetadataCreator(root)
    root.mainloop()
