from typing import Dict
import os
import subprocess
import json

# TODO: Confirm all options are storing and rehydrating

def write(video_mash):
    source_paths = [os.path.abspath(source.source_path) for source in video_mash.selected_sources]
    starting_frames = [source.starting_frame for source in video_mash.selected_sources]
    effects = ','.join(video_mash.effects) if video_mash.effects and len(video_mash.effects) > 0 else ''
    metadata = {
        "source_paths": ",".join(source_paths),
        "starting_frames": ",".join(map(str, starting_frames)),
        "seconds": video_mash.seconds,
        "width": video_mash.width,
        "height": video_mash.height,
        "mode": video_mash.mode,
        "effects": effects,
        "fps": video_mash.fps,
        "colormap": video_mash.colormap,
        "script_version": __version__ if '__version__' in globals() else "Unknown"
    }

    save_path = video_mash.output_path.with_suffix('.metadata' + video_mash.output_path.suffix)

    metadata_args = []
    for k, v in metadata.items():
        metadata_args.extend(['-metadata', f'{k}={v}'])

    args = [
        'ffmpeg',
        '-v', 'quiet',
        '-i', str(video_mash.output_path.absolute()),
        '-movflags',
        'use_metadata_tags',
        # '-map_metadata', '0',
        *metadata_args,
        '-c', 'copy',
        str(save_path.absolute())
    ]

    try:
        # Run ffmpeg command
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace the original file with the new one
        os.replace(str(save_path), str(video_mash.output_path))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running ffmpeg: {e.stderr.decode()}") from e
    finally:
        # Delete the save file if it still exists
        if save_path.exists():
            save_path.unlink()

def read(file_path):
    try:
        metadata = _read_raw_metadata(file_path)

        if metadata:
            metatags = metadata.get('tags')
            print(metadata)
            if metatags:
                mash_data = {}
                mash_data['source_paths'] = metatags.get('source_paths').split(',')
                mash_data['starting_frames'] = list(map(int, metatags.get('starting_frames').split(',')))
                mash_data['seconds'] = float(metatags.get('seconds'))
                mash_data['width'] = int(metatags.get('width'))
                mash_data['height'] = int(metatags.get('height'))
                mash_data['effects'] = list(map(str, metatags.get('effects').split(',')))
                mash_data['fps'] = float(metatags.get('fps'))
                mash_data['colormap'] = metatags.get('colormap')
                # Parsing out extra characters from when mode was an array
                mash_data['mode'] = metatags.get('mode').replace("'", "").strip("[]")

                return mash_data
            else:
                return False
        else:
            return False
    except Exception as e:
        print(f"!!! exception {e}")
        return False

def _read_raw_metadata(file_path) -> Dict:
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-print_format', 'json',
        '-show_format',
        file_path
    ]

    try:
        # Run ffprobe command and capture output
        result = subprocess.run(ffprobe_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_json = result.stdout.decode('utf-8')
        metadata_dict = json.loads(output_json)['format']
        return metadata_dict
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running ffprobe: {e.stderr.decode()}") from e
