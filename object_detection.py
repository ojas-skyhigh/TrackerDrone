import os
import time
from ultralytics import YOLO
import supervision as sv
import pyrealsense2 as rs
import numpy as np
import cv2
import math 


def init_tracker():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'v2/runs/detect/train/weights/best.pt') 

    model = YOLO(model_path)
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    return model, pipeline, align

def stream_obj_coords(model, pipeline, align):
    last_cx, last_cy = 320, 240
    last_cz = None
    last_good_coords = None
    
    grace_frames = 0
    MAX_GRACE = 5
    is_tracking = False       # Flag to allow the very first detection to bypass jump filters
    MAX_PX_JUMP = 150         # Max allowed pixel jump per frame
    MAX_Z_JUMP = 0.4          # Max allowed Z-depth jump per frame (meters)
    
    try:
        while True:
            start_time = time.time() 
            
            frames = pipeline.wait_for_frames(5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            
            results = model.track(color_image, device='cuda', conf=0.55, verbose=False, persist=True)

            detections = sv.Detections.from_ultralytics(results[0])
            detections = detections.with_nms(threshold=0.4)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=color_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            height, width = color_image.shape[:2]
            current_coordinates = None
            valid_detection_this_frame = False

            if len(detections.xyxy) > 0:
                best_idx = 0
                min_distance = float('inf')
                
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = map(int, detections.xyxy[i])
                    test_cx = int((x1 + x2) / 2)
                    test_cy = int((y1 + y2) / 2)
                    pixel_dist = math.hypot(test_cx - last_cx, test_cy - last_cy)
                    if pixel_dist < min_distance:
                        min_distance = pixel_dist
                        best_idx = i

                x1, y1, x2, y2 = map(int, detections.xyxy[best_idx])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                valid_distances = []
                patch_size = 2  
                for dy in range(-patch_size, patch_size + 1):
                    for dx in range(-patch_size, patch_size + 1):
                        px, py = cx + dx, cy + dy
                        if 0 <= px < width and 0 <= py < height:
                            dist = depth_frame.get_distance(px, py)
                            if dist > 0.0: 
                                valid_distances.append(dist)

                if valid_distances:
                    cz = np.median(valid_distances)
                    
                    is_valid_move = True

                    if cz < 0.3 or cz > 3.2:
                        is_valid_move = False
                        cv2.putText(annotated_image, "REJECTED: DEPTH OUT OF RANGE", (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if is_valid_move and is_tracking and last_cz is not None:
                        px_jump = math.hypot(cx - last_cx, cy - last_cy)
                        z_jump = abs(cz - last_cz)
                        
                        if px_jump > MAX_PX_JUMP or z_jump > MAX_Z_JUMP:
                            is_valid_move = False
                            cv2.putText(annotated_image, "REJECTED: SPEED JUMP", (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if is_valid_move:
                        last_cx, last_cy, last_cz = cx, cy, cz
                        is_tracking = True
                        grace_frames = 0
                        valid_detection_this_frame = True
                        current_coordinates = (cx, cy, cz)
                        last_good_coords = current_coordinates
                        
                        cv2.circle(annotated_image, (cx, cy), radius=6, color=(0, 0, 255), thickness=-1)
                        cv2.putText(annotated_image, f"{cz:.2f}m", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_image, "DEPTH BLIND", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    is_valid_move = False


            if not valid_detection_this_frame:
                if grace_frames < MAX_GRACE and last_good_coords is not None:
                    grace_frames += 1
                    current_coordinates = last_good_coords
                    cv2.putText(annotated_image, f"COASTING ({grace_frames}/{MAX_GRACE})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    is_tracking = False  # Reset filter lock so next detection is trusted
                    last_good_coords = None
                    current_coordinates = None
                    cv2.putText(annotated_image, "TARGET LOST", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- CALCULATE & DRAW LATENCY ---
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            cv2.putText(annotated_image, f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("PlutoX Tracker", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
            
            yield current_coordinates
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()