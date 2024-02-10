import sys
print(sys.path)
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GES', '1.0')
from gi.repository import Gst, GES
import glob

def main():
    # Check if a file glob is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_glob>")
        sys.exit(1)

    file_glob = sys.argv[1]

    # Initialize GStreamer
    Gst.init(None)

    # Create a GES timeline
    timeline = GES.Timeline.new_audio_video()

    # Add source videos to the timeline
    for source_uri in glob.glob(file_glob):
        source_clip = GES.UriClipAsset.request_sync(f"file://{source_uri}")
        timeline.add_clip(source_clip)

        # Resize and crop each clip to 1280x720
        for track in source_clip.get_video_tracks():
            caps = track.get_caps().get_structure(0)
            caps.set_value("width", 1280)
            caps.set_value("height", 720)
            caps.set_value("crop-left", 0)
            caps.set_value("crop-right", 0)
            caps.set_value("crop-top", 0)
            caps.set_value("crop-bottom", 0)

    # Add a new layer to the timeline
    layer = GES.Layer.new()
    timeline.add_layer(layer)

    # Add clips to the layer
    for clip in timeline.get_clips():
        layer.add_clip(clip)

    # Apply blend mode (soft_light) to the layer
    layer.set_blending_mode(GES.BlendMode.SOFT_LIGHT)

    # Render the timeline
    pipeline = timeline.create_pipeline()

    # Create and configure filesink
    output_file = "output.mp4"
    sink = Gst.ElementFactory.make("filesink", "filesink")
    sink.set_property("location", output_file)
    pipeline.add(sink)
    pipeline.get_by_name("videoconvert0").link(sink)

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    # Wait until the end of the stream
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)

    # Release resources
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
