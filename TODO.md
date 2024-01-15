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

