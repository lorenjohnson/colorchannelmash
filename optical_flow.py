import cv2
import numpy as np

def get_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_list.append(flow)

        prev_gray = gray

    cap.release()
    return flow_list

def apply_optical_flow_to_frame(frame, flow):
    # Apply optical flow to the frame
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    displacement = flow_map + flow.reshape(-1, 2)
    displacement = np.clip(displacement, 0, (h - 1, w - 1))

    warped_frame = cv2.remap(frame, displacement[:, 1].astype(np.float32), displacement[:, 0].astype(np.float32), cv2.INTER_LINEAR)

    return warped_frame

def process_frame(frame):
    # Your existing frame processing logic goes here
    # This function should take a frame and perform any desired processing
    # For example, you might apply filters, transformations, etc.
    # Return the processed frame
    pass

def composite_frames(target_frame, water_frame):
    # Simple compositing example (you can customize this based on your needs)
    result_frame = cv2.addWeighted(target_frame, 0.7, water_frame, 0.3, 0)
    return result_frame

def main():
    water_flow = get_optical_flow('water_video.mp4')
    cap_target = cv2.VideoCapture('target_video.mp4')

    while True:
        ret, target_frame = cap_target.read()
        if not ret:
            break

        # Process the target frame (e.g., apply filters)
        processed_target_frame = process_frame(target_frame)

        # Apply water motion to the frame
        flow_index = min(cap_target.get(cv2.CAP_PROP_POS_FRAMES), len(water_flow)-1)
        water_flow_frame = water_flow[int(flow_index)]
        water_motion_frame = apply_optical_flow_to_frame(processed_target_frame, water_flow_frame)

        # Composite the frames
        result_frame = composite_frames(processed_target_frame, water_motion_frame)

        # Display or save the resulting frame
        cv2.imshow('Montage', result_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap_target.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
