
LIVE
  Make webcam a live source during video rendering
  Make video sources play on top of each other in layer selection, starting from the start frame of when each is selected

Bring back audio for all clips

Add still images animation: Decay them over time, extending the current colorchannelmash script?

Now that I can regnerate from a mash:

- Make sure that CLI params override things in the file
- Maybe only run once if there is one file, ignore numSets (or just make numSets go away)

Add Video Moshing function

Try again accessing Apple Photos (and bonus if in more than 1 library at once)

Get a custom LUT working (vs just the build in ones)



I want the user to be able to cycle through possible selections for the 3 sources they are going to use for the video to be rendered before it starts. We can just present them with 1 randomly selected source video at a randomly selected start frame, and they can press (enter) if they like it, or (n) to b presented with another option. They do until they have selected 3 options:


before rendering video randomly pick the first source and start frame,
present this to the user. user can pres
(2) to cycle between options for source, (enter) to select (this happens 3 times until all 3 sources are chosen)
(c) cycle color space options on preview


during render:
(g) add 10 glitch frames in a row now
(m) mos next 3 seconds of frames
Extract source files, starting frames, and settings from a file  (a mash)
Run for -i or --input file.avi:frame_num or --input file.avi (picks random starting frame)

Option to provide file list / Make interactive choices of source images / Make reproducible images
Interactive colorSpace switch
Animate movement of large images
Try on other sources
Fix glob command line source param so it doesn't need quotes



Can you add to this python script take command line params as follows, updating the logic accordingly:

<sourceGlob> (positional. a file path glob that can either be just a directory or a glob pattern, e.g. source/*.mov. optional, defaults to "source/*.(mov|avi|mp4)")
--numSets (total number of sets to generate. optional, defaults to 1)
--setLength <seconds> (optional, defaults to 10 seconds)
--width <number> (output video width. optional, defaults to iPhone 11 Pro Max screen width)
--height <number> (output video height. optional, defaults to iPhone 11 Pro Max screen height)
--outputDir (optiona, defaults to 'output')
--fps <number> (optional, defaults to 30)

Things to note:
- In this version a new file is written for each set instead of one continual file. Each set file goes into the outputDir and is named like "set-001.avi" where 001 is the set number. Also the set number starts as one more than the latest highest numbered set file found in the output directory.
- The output video is no longer the max height width of the source videos but manually specified by the width and height params. Source videos/frames should simple crop to this size and fill in the rest of the frame with 0s when the frame or the crop is smaller than the target height and width.




Upcoming: 
- Take from another file if one errors on initial load of the 3, will give a more consistelnly rich result
- Render from more focused sets (current week for instance for Insta posting)
- Cropping vs getting max frame size
- Try to render with HSV color space vs RGB
- Crop one frame from the middle, another from the top and another from the bottom, where applicable
- Add some contrast to each channel by default
- Optional effect "hooks": 
  - set_effect (applies to entire set at the end of frame-by-frame-processing)
  - channel_effect (applies to all channels in the set)
  - green_channel_effect (applies to green channel in the entire set)
  - red_channel_effect
  - blue_channel_effect
  applies randomly to each set channel (or set result), if colorchannelmash_effects.py is available and has set_channel_effect or set_effect present, respectively. set_channel_effect is applied to only one channel for each set.

