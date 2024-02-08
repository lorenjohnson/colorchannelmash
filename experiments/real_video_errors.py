import random
# import cv2
# import argparse
# import os
# import glob
import osxphotos
# from pathlib import Path

def introduce_binary_errors(input_file, output_file, error_probability=0.01, max_error_length=100):
    with open(input_file, 'rb') as input_stream:
        video_data = bytearray(input_stream.read())

        for _ in range(int(len(video_data) * error_probability)):
            # Randomly select a position to start the error
            start_position = random.randint(0, len(video_data) - 1)

            # Randomly determine the length of the error
            error_length = random.randint(1, max_error_length)

            # Randomly modify the binary data in the selected range
            for i in range(error_length):
                if start_position + i < len(video_data):
                    video_data[start_position + i] = random.randint(0, 255)

    with open(output_file, 'wb') as output_stream:
        output_stream.write(video_data)

def get_apple_photos_movies(dbfile=None, albums=None):
    if dbfile:
        photosdb = osxphotos.PhotosDB(dbfile=dbfile)
    else:
        photosdb = osxphotos.PhotosDB()

    videos = photosdb.photos(images=False, movies=True, albums=albums)

    # Filter out None results
    videos = [video for video in videos if video is not None]
    paths = []

    for video in videos:
        path = None
        if video.hasadjustments:
            path = video.path_edited
        else:
            path = video.path
        if path is not None:
            paths.append(path)

    return paths

if __name__ == "__main__":
    some_movies = get_apple_photos_movies()
    random_source_path = random.choice(some_movies)
    introduce_binary_errors(random_source_path, 'test.mp4')
  
