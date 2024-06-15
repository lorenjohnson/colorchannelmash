import argparse
import os
import subprocess
import glob
import cv2
import osxphotos
from pathlib import Path
from video_mash import VideoMash, ExitException, BLEND_MODES, EFFECTS, EFFECT_COMBOS
from video_mash.video_mash_metadata import read as read_metadata

class VideoMashCLI:
    def __init__(self):
        self.args = self.parse_args()
        self.source_paths = []
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Video Masher: A memory machine.")
        parser.add_argument("--osxphotos", action='store', default=None, nargs='?', const=True,
                            help="Use movies in the current Apple Photos Library as source (MacOS only). "
                                "If a file path follows, it will be used as the Apple Photos Library path.")
        parser.add_argument("--albums", nargs='+', default=None, 
                            help="List of album names to use. If not specified, it will use the last album or all photos. (Apple Photos only)")
        parser.add_argument("sourceGlob", nargs='*', default=["source/*.mov"],
                            help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.mov'.")
        parser.add_argument("--outputDir", default="output",
                            help="Output directory for mash files. Optional, defaults to 'output'.")
        parser.add_argument("--mashFile",
                            help="Either a previously rendered mash that you want to base a new render on, or a JSON mash composition.")
        parser.add_argument("--seconds", type=int, default=10,
                            help="Duration of each mash in seconds. Optional, defaults to 10 seconds.")
        parser.add_argument("--width", type=int, default=1242,
                            help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
        parser.add_argument("--height", type=int, default=2688,
                            help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
        parser.add_argument("--mode", choices=BLEND_MODES, default='multiply',
                            help="Set the mash mode, can be combined with an amount param (0-0.99), e.g. \"--mode soft-light:0.7\". Optional, defaults to channels.")
        parser.add_argument("--opacity", type=float, default=0.5,
                            help="Set opacity for the foreground layers for the blend mode. Optional, defaults to 0.5.")
        parser.add_argument("--effects", nargs='+', choices=EFFECTS, default=EFFECT_COMBOS[0],
                            help="Set effect(s) to apply to each frame. Options: hsv, hls, yuv, gray, invert, ocean. Optional, defaults to None.")
        parser.add_argument("--brightness", type=float, default=None,
                            help="Set a target brightness for each frame. Options: 0 to 0.9, Optional, defaults to no change.")
        parser.add_argument("--contrast", type=float, default=None,
                            help="Set a target contrast for each frame. Options: 0 to 0.9, Optional, defaults to no change.")
        parser.add_argument("--fps", type=int, default=12,
                            help="Frames per second for output videos. Optional, defaults to 30.")
        parser.add_argument("--mashes", type=int, default=10,
                            help="Total number of mashes to generate. Rendering preview is off if this is set. Optional, defaults to 10")
        parser.add_argument("--auto", type=int, default=1,
                            help="Automatically setup mashes with this many randomly selected layers")
        return parser.parse_args()

    def check_ffmpeg_installed(self):
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.") from e

    def get_apple_photos_movies(self, dbfile=None, albums=None):
        if dbfile:
            photosdb = osxphotos.PhotosDB(dbfile=dbfile)
        else:
            photosdb = osxphotos.PhotosDB()
        videos = photosdb.photos(images=False, movies=True, albums=albums)
        videos = [video for video in videos if video is not None]
        paths = []
        for video in videos:
            path = video.path_edited if video.hasadjustments else video.path
            if path:
                paths.append(path)
        return paths

    def gather_source_paths(self):
        if self.args.osxphotos is True:
            self.source_paths = self.get_apple_photos_movies(albums=self.args.albums)
        elif self.args.osxphotos:
            self.source_paths = self.get_apple_photos_movies(self.args.osxphotos, self.args.albums)
        else:
            if len(self.args.sourceGlob) == 1:
                source = self.args.sourceGlob[0]
                if source.endswith('.txt'):
                    with open(source, 'r') as file:
                        self.source_paths = [line.strip() for line in file if line.strip()]
                elif os.path.isfile(source) and source.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    metadata = read_metadata(source)
                    if metadata and 'source_paths' in metadata:
                        self.source_paths = metadata['source_paths']
                    else:
                        self.source_paths = [source]
                else:
                    self.source_paths = glob.glob(source)
            else:
                for source_glob in self.args.sourceGlob:
                    self.source_paths.extend(glob.glob(source_glob))

    def run(self):
        self.check_ffmpeg_installed()
        self.gather_source_paths()
        
        if not self.source_paths:
            print(f"No video files found using the provided sourceGlob: {self.args.sourceGlob}")
            return
        
        try:
            print(f"{len(self.source_paths)} source videos found")
            existing_mashes = glob.glob(os.path.join(self.args.outputDir, 'mash-*'))
            max_mash_number = 0
            if existing_mashes:
                max_mash_number = max(int(s.split('-')[1].split('.')[0]) for s in existing_mashes)
            mash_number = max_mash_number + 1

            while mash_number <= max_mash_number + self.args.mashes:
                output_path = Path(os.path.join(self.args.outputDir, f"mash-{mash_number:03d}.mp4"))

                print(f"Starting video mash {mash_number}...")

                video_mash = VideoMash(
                    source_paths=self.source_paths,
                    output_path=output_path,
                    mash_file=self.args.mashFile,
                    mode=self.args.mode,
                    opacity=self.args.opacity,
                    effects=self.args.effects,
                    brightness=self.args.brightness,
                    contrast=self.args.contrast,
                    seconds=self.args.seconds,
                    fps=self.args.fps,
                    width=self.args.width,
                    height=self.args.height,
                    auto=self.args.auto
                )
                success = video_mash.mash()

                if success:
                    mash_number += 1
        except ExitException as e:
            cv2.destroyAllWindows()
            if e: print(f"{e}")
            print("Bye!")

if __name__ == "__main__":
    cli = VideoMashCLI()
    cli.run()
