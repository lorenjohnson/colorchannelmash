import argparse
import os
import subprocess
import glob
import cv2
from pathlib import Path
from video_mash import VideoMash, ExitException

def parse_args():
    parser = argparse.ArgumentParser(description="Video montage script with command line parameters.")
    parser.add_argument("sourceGlob", nargs='*', default=["source/*.mov"],
                        help="File path glob for source videos (e.g., source/*.mov). Optional, defaults to 'source/*.mov'.")
    parser.add_argument("--outputDir", default="output", help="Output directory for mash files. Optional, defaults to 'output'.")
    parser.add_argument("--seconds", type=int, default=10, help="Duration of each mash in seconds. Optional, defaults to 10 seconds.")
    parser.add_argument("--width", type=int, default=1242, help="Output video width. Optional, defaults to iPhone 11 Pro Max screen width.")
    parser.add_argument("--height", type=int, default=2688, help="Output video height. Optional, defaults to iPhone 11 Pro Max screen height.")
    parser.add_argument("--colorSpace", choices=['rgb', 'hsv', 'hls', 'yuv', 'gray'], default='rgb',
                        help="Color space for displaying and combining frames. Options: hsv, hls, rgb, yuv, gray. Optional, defaults to rgb.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos. Optional, defaults to 30.")
    parser.add_argument("--mashes", type=int, default=10,
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

    for source_glob in args.sourceGlob:
        source_paths.extend(glob.glob(source_glob))

    if not source_paths:
        print(f"No video files found using the provided sourceGlob: {args.sourceGlob}")
    else:
        try:
            existing_mashes = glob.glob(os.path.join(args.outputDir, 'mash-*'))
            max_mash_number = 0
            if existing_mashes:
                max_mash_number = max(int(s.split('-')[1].split('.')[0]) for s in existing_mashes)
            mash_number = max_mash_number + 1

            while mash_number <= max_mash_number + mash_number:
                output_path = os.path.join(args.outputDir, f"mash-{mash_number:03d}.mp4")

                print(f"Starting video mash {mash_number}...")

                video_mash = VideoMash(
                    source_paths, args.seconds, args.width, args.height, args.colorSpace, args.fps
                )
                success = video_mash.mash(output_path)

                if success: mash_number += 1
        except ExitException as e:
            cv2.destroyAllWindows()
            if e: print(f"{e}")
            print("Bye!")

if __name__ == "__main__":
    main()
