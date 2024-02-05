import os
import tempfile
import argparse

import cv2
from alive_progress import alive_bar

def interpolate_frames(input_file, output_file, num_interpolated_frames, provided_fps, overwrite):
    if not output_file:
        output_file = input_file

    # Use a temporary file if no output file is designated
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output_file = temp_output.name
        
    # Check if the output file already exists and confirmation is required
    if os.path.exists(output_file) and not overwrite:
        user_input = input(f"Output file '{output_file}' already exists. Do you want to overwrite it? (y/n): ")
        if user_input.lower() != 'y':
            print("Interpolation canceled.")
            return

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new frames per second after interpolation
    new_fps = provided_fps if provided_fps else fps * (num_interpolated_frames + 1)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_file, fourcc, new_fps, (frame_width, frame_height))

    ret, frame = cap.read()

    with alive_bar(total_frames) as bar:
        while ret:
            # Write original frame
            out.write(frame)
            # Update progress bar

            # Interpolate frames
            ret, next_frame = cap.read()
            for _ in range(num_interpolated_frames):
                if not ret:
                    break

                # Calculate interpolated frame
                interpolated_frame = cv2.addWeighted(frame, 0.5, next_frame, 0.5, 0)
                out.write(interpolated_frame)
                # Update progress bar

                # Update frame for the next iteration
                frame = interpolated_frame

            frame = next_frame
            # out.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
            bar()

    # Release video capture and writer objects
    cap.release()
    out.release()
    os.replace(temp_output_file, output_file)

def main():
    parser = argparse.ArgumentParser(description='Interpolate frames in a video file.')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('output_file', nargs='?', help='Output video file path (optional)')
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to interpolate between each original frame')
    parser.add_argument('--fps', type=int, default=False, help='Force an FPS on output file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file without confirmation')

    args = parser.parse_args()

    # Call the interpolation function
    interpolate_frames(args.input_file, args.output_file, args.frames, args.fps, args.overwrite)
    if args.output_file and args.overwrite:
        print(f"Interpolation completed. Output saved to {args.output_file}")
    elif args.output_file and not args.overwrite:
        print(f"Interpolation completed. Output saved to {args.output_file}")
    else:
        print("Interpolation completed.")

if __name__ == "__main__":
    main()
