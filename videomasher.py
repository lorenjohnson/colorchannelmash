import argparse
import os
import subprocess
import glob
import cv2
import osxphotos
from pathlib import Path
from video_mash import VideoMash, ExitException

def parse_args():
    parser = argparse.ArgumentParser(description="Video Masher: A memory machine.")
    parser.add_argument(
        "--osxphotos",
        action='store',
        default=None,
        nargs='?',
        const=True,
        help="Use movies in the current Apple Photos Library as source (MacOS only). "
             "If a file path follows, it will be used as the Apple Photos Library path.")
    parser.add_argument(
        "sourceGlob",
        nargs='*',
        default=["source/*.mov"],
        help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.mov'.")
    parser.add_argument(
        "--outputDir",
        default="output",
        help="Output directory for mash files. Optional, defaults to 'output'.")
    parser.add_argument(
        "--mashFile",
        help="Either a previously rendered mash that you want to base a new render on, or a JSON mash composition.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=10,
        help="Duration of each mash in seconds. Optional, defaults to 10 seconds.")
    parser.add_argument(
        "--width",
        type=int,
        default=1242,
        help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
    parser.add_argument(
        "--height",
        type=int,
        default=2688,
        help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
    parser.add_argument(
        "--mode",
        nargs='+',
        choices=['channels', 'accumulate', 'soft_light', 'lighten_only', 'dodge', 'addition', 'darken_only', 'multiply', 'hard_light',
            'difference', 'subtract', 'grain_extract', 'grain_merge', 'divide', 'overlay', 'normal'],
        default=['channels'],
        help="Set the mash mode. Optional, defaults to multiply.")
    parser.add_argument(
        "--effects",
        nargs='+',
        choices=['hsv', 'hls', 'yuv', 'gray', 'invert', 'ocean'],
        default=[],
        help="Set which effect(s) to apply to each frame. Currently just colormodes. Options: hsv, hls, rgb, yuv, gray. Optional, defaults to None.")
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="Set a target brightness for each frame. Options: 0 to 0.9, Optional, defaults to no change.")
    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="Set a target contrast for each frame. Options: 0 to 0.9, Optional, defaults to no change.")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos. Optional, defaults to 30.")
    # TODO: Default to loop, then this setting becomes a limit
    parser.add_argument(
        "--mashes",
        type=int,
        default=10,
        help="Total number of mashes to generate. Rendering preview is off if this is set. Optional, defaults to 10.")

    return parser.parse_args()

def main():
    # Check if ffmpeg is installed, no mashing without it (needed for storing new mashes and regenerating old ones)
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.") from e

    args = parse_args()
    source_paths = []

    if args.osxphotos is True:
        source_paths = get_apple_photos_movies()
    elif args.osxphotos is not None:
        source_paths = get_apple_photos_movies(args.osxphotos)
    else:
        for source_glob in args.sourceGlob:
            source_paths.extend(glob.glob(source_glob))

    if not source_paths:
        print(f"No video files found using the provided sourceGlob: {args.sourceGlob}")
    else:
        try:
            print(f"{len(source_paths)} source videos found")
            existing_mashes = glob.glob(os.path.join(args.outputDir, 'mash-*'))
            max_mash_number = 0
            if existing_mashes:
                max_mash_number = max(int(s.split('-')[1].split('.')[0]) for s in existing_mashes)
            mash_number = max_mash_number + 1

            while mash_number <= max_mash_number + args.mashes:
                output_path = Path(os.path.join(args.outputDir, f"mash-{mash_number:03d}.mp4"))

                print(f"Starting video mash {mash_number}...")

                video_mash = VideoMash(
                    source_paths=source_paths,
                    output_path=output_path,
                    mash_file=args.mashFile,
                    mode=args.mode,
                    effects=args.effects,
                    brightness=args.brightness,
                    contrast=args.contrast,
                    seconds=args.seconds,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                )
                success = video_mash.mash()

                if success: mash_number += 1
        except ExitException as e:
            cv2.destroyAllWindows()
            if e: print(f"{e}")
            print("Bye!")

def get_apple_photos_movies(dbfile = None):
    if dbfile:
        photosdb = osxphotos.PhotosDB(dbfile=dbfile)
    else:
        photosdb = osxphotos.PhotosDB()

    movies = photosdb.photos(images=False, movies=True)
    
    # Filter out None results
    movies = [movie for movie in movies if movie is not None]
    paths = []
    
    for movie in movies:
        path = None
        if movie.hasadjustments:
          path = movie.path_edited
        else:
          path = movie.path
        if (path is None):
          print(f"ismissing: {movie.ismissing}, isreference: {movie.isreference}, hidden: {movie.hidden}, shared: {movie.shared}")
        else:
          paths.append(path)
          
    return paths

if __name__ == "__main__":
    main()
