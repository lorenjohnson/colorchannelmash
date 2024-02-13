import sys
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
from pynput import keyboard
import subprocess
import os
import random

def on_press(key):
    if key == keyboard.Key.esc:
        print("Rendering ended by pressing ESC.")
        os.system("pkill afplay")  # Stop audio playback
        return False

def create_multivideo(video_files, output_file):
    # Shuffle the list of video files
    random.shuffle(video_files)
    
    # Load video clips
    clips = [VideoFileClip(file, audio=False) for file in video_files]
    
    # Calculate maximum duration
    max_duration = max([clip.duration for clip in clips])
    
    # Create a list to store the clips for the final video
    final_clips = []
    
    # Loop to create the final video
    for i in range(int(max_duration)):
        # Calculate the time index in the clip sequence
        time_index = i % len(clips)
        
        # Add the clips at the current time index to the final clips list
        final_clips.append(clips_array([[clips[time_index], clips[(time_index + 1) % len(clips)], clips[(time_index + 2) % len(clips)]]]))
        
        # Concatenate the final clips
        final_clip = concatenate_videoclips(final_clips)
        
        # Write a temporary video file for preview
        # temp_file = "temp_preview.mp4"
        # final_clip.write_videofile(temp_file, fps=24, audio=False, rewrite_audio=False)
        
        # # Check if the temporary video file exists before playing it
        # if os.path.exists(temp_file):
        #     # Play the temporary video file in a separate process
        #     subprocess.Popen(['afplay', temp_file])
        
        #     # Listen for Esc key to end rendering
        #     with keyboard.Listener(on_press=on_press) as listener:
        #         listener.join()
        
        #     # Remove the temporary video file
        #     os.remove(temp_file)
        # else:
        #     print("Error: Temporary video file does not exist.")

    # Write the final video to a file
    final_clip.write_videofile(output_file, rewrite_audio=False, audio=False, fps=24)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_glob> <output_file>")
        sys.exit(1)

    file_glob = sys.argv[1]
    output_file = sys.argv[2]

    video_files = glob.glob(file_glob)
    create_multivideo(video_files, output_file)
