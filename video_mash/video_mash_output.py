import os
import subprocess
import shlex
import json

class VideoMashOutput:
    def __init__(self, source_path):
        self.source_path = source_path
        self.metadata = self.read_metadata(source_path)

    def write_metadata(self, output_path, seconds, width, height, color_space, fps):
        output_path = Path(output_path)
        source_path = os.path.abspath(self.source_path)

        metadata = {
            "source_paths": source_path,
            "starting_frames": 0,
            "seconds": seconds,
            "width": width,
            "height": height,
            "color_space": color_space,
            "fps": fps,
            "script_version": __version__ if '__version__' in globals() else "Unknown"
        }

        save_path = output_path.with_suffix('.metadata' + output_path.suffix)

        metadata_args = []
        for k, v in metadata.items():
            metadata_args.extend(['-metadata', f'{k}={v}'])

        args = [
            'ffmpeg',
            '-v', 'quiet',
            '-i', shlex.quote(str(output_path.absolute())),
            '-movflags',
            'use_metadata_tags',
            # '-map_metadata', '0',
            *metadata_args,
            '-c', 'copy',
            shlex.quote(str(save_path))
        ]

        try:
            # Run ffmpeg command
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Replace the original file with the new one
            os.replace(save_path, output_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running ffmpeg: {e.stderr.decode()}") from e
        finally:
            # Delete the save file if it still exists
            if os.path.exists(save_path):
                os.remove(save_path)
