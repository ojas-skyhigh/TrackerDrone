import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

def capture_raw_dataset():
    # Creates a unique folder for every single time you run the script (e.g., dataset_raw_1718000000)
    # This guarantees you will NEVER overwrite your previous images.
    session_name = f"dataset_raw_{int(time.time())}"
    os.makedirs(session_name, exist_ok=True)
    print(f"Saving raw images to folder: {session_name}/")

    # Initialize RealSense Camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    frame_counter = 0
    saved_count = 0
    save_interval = 10  # Captures 1 out of every 10 frames

    print("\nRecording started. Press 'q' to stop recording.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Clone the image for the UI so the saved image remains 100% untouched
            display_image = color_image.copy()
            frame_counter += 1

            # Save exactly 1 in 10 frames
            if frame_counter % save_interval == 0:
                # Use zero-padded counting to keep files ordered chronologically
                img_path = os.path.join(session_name, f"frame_{saved_count:05d}.jpg")
                
                # Save the completely raw, untouched image
                cv2.imwrite(img_path, color_image)
                
                # Draw a red dot on the display window ONLY so you know it's capturing
                cv2.circle(display_image, (32, 32), 12, (0, 0, 255), -1)
                saved_count += 1

            # UI Overlay (Does not affect the saved images)
            cv2.putText(display_image, f"Saved Raw Frames: {saved_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Raw Frame Capture", display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nSuccessfully saved {saved_count} raw images to {session_name}/")

if __name__ == "__main__":
    capture_raw_dataset()
