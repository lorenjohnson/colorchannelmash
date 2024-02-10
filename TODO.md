Feature: Optionally allow each layer to have its own blend mode and opacity
Bug: Back out of a render to recompose. Crashes on metadata write, but probably because file doesn't close properly
Feature: Make window full screen with F

Feature: Save last settings used and use as defaults on next run. OR Save Settings?

Feature: Look into mashfile starting frame inconsistency? 
Feature: Make mashfile restore more robust to failures of effects and blend modes...
Feature: Ctrl-C asks to save file?
Feature: Explore keep_color_channels_separated color reduction as one stage of alchemic transmformation (for instance can compose more images in later generations once colors have been reduced in individual mashes / images)

Feature: Save Mash data in screenshot file as well
Feature: GUI Show current settings and button to "save mash"
Feature: Automatic mode... using effect hls and mode channels. Think about non-interactive screens... TV watching. DMN
Feature: Multiple OSX photos libraries? (confirm multiple source directories already)

Feature: Option to specify a base layer

Video specific:
Feature: See how I might speed-up the other blend_modes besides channels...
Feature: Explore degredation algorithm within a video render, such that it start re-using frames it has already rendered in composite with later frames...
Feature: Keep audio through pre-processing, and combine in output (Move to MoviePy for pre-processing? Or not. Ask CGPT)
Feature: Re-random start_frame if the one chosen is within the last 80% of the video?
Feature: Clips no longer loop past their total length. new clips are brought in.
Feature: Bring in a new clip when one layer runs out of length (but still let it loop to the total length of the clip)
Feature: + / - add remove to length of currently rendering video

Bug: Back/forward from render state still sometimes index goes out of range
Feature: Moshing / more creative destruction chaos to files
Feature: Mash files...
Feature: When hitting one Esc in the midst of rendering make it possible to start re-selecting layers again by simply hitting spacebar, throwing out the current render (partially implemented)
Feature: Bring back audio for all clips
Feature: Make --effect cli param take color map options...
Feature: Add mode cycling during select stage ("m")
Feature: Add effect(s) cycling during select stage ("e")
Feature: Change colorSpace CLI arg to colorMode
  rgb (default) gray, hsl, yuv, map_ocean (etc, cv2 builtins), <custom_lut_filepath>
Feature: Change color modes on the fly
Feature: Add still images animation: Decay them over time, extending the current colorchannelmash script?

Feature: Make stand-alone installer / runner / release path

LIVE
  How to speed-up rendering
  Make webcam a live source during video rendering
  Make video sources play on top of each other in layer selection, starting from the start frame of when each is selected
  Have it dynamically take-in new sources as they are found

Sound track random selection / keyboard selection from source directories (wav, mp3, aif)
Reverse playback randomly, or make it an option

Try with just images

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
- Crop one frame from the middle, another from the top and another from the bottom, where applicable
- Add some contrast to each channel by default
- Optional effect "hooks": 
  - set_effect (applies to entire set at the end of frame-by-frame-processing)
  - channel_effect (applies to all channels in the set)
  - green_channel_effect (applies to green channel in the entire set)
  - red_channel_effect
  - blue_channel_effect
  applies randomly to each set channel (or set result), if colorchannelmash_effects.py is available and has set_channel_effect or set_effect present, respectively. set_channel_effect is applied to only one channel for each set.

