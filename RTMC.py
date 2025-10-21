import cv2
import numpy as np
import time
import math

def start_motion_detection():
    motion_count = 0 
    tracked_objects = {}
    next_id = 0 
    history_buffer = {} 
    min_contour_area = 900 
    max_match_distance = 120 
    min_movement_threshold = 8
    arrow_scale = 35
    arrow_smoothing = 15
    rapid_movement_threshold = 25
    min_aspect_ratio = 0.3
    max_aspect_ratio = 2.5
    HEAD_COLOR = (255, 0, 0)
    ARM_COLOR = (0, 255, 0)
    RAPID_COLOR = (0, 0, 255)
    ARROW_COLOR = (0, 0, 255)
    try:
        print("Starting camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        _, first_frame = cap.read()
        if first_frame is None:
            print("Error: Could not read from camera")
            cap.release()
            return
            
        frame_height, frame_width = first_frame.shape[:2]
        print(f"Camera resolution: {frame_width}x{frame_height}")
        

        ret, frame1 = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            cap.release()
            return
        
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Could not read second frame from camera")
            cap.release()
            return

        print("Human tracking active. Press 'q' to quit.")
        print("BLUE: Head | GREEN: Arms | RED: Rapid movements | RED ARROWS: Direction")
        
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        frame_skip = 1  
        frame_counter = 0
        
        debug_mode = True
        
        while True:
            frame_count += 1
            current_time = time.time()
            if (current_time - prev_time) >= 1.0:
                fps = frame_count
                frame_count = 0
                prev_time = current_time
            
            frame_counter += 1
            if frame_skip > 1 and frame_counter % frame_skip != 0:
                frame1 = frame2
                ret, frame2 = cap.read()
                if not ret:
                    print("Error reading frame, exiting")
                    break
                continue
            
            frame_vis = frame2.copy()
            
            try:
                diff = cv2.absdiff(frame1, frame2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh, None, iterations=2)
                
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if debug_mode:
                    small_dilated = cv2.resize(dilated, (frame_width // 4, frame_height // 4))
                    small_dilated = cv2.cvtColor(small_dilated, cv2.COLOR_GRAY2BGR)
                    frame_vis[0:frame_height//4, 0:frame_width//4] = small_dilated
            except Exception as e:
                print(f"Error processing frames: {str(e)}")
                frame1 = frame2
                ret, frame2 = cap.read()
                if not ret:
                    print("Error reading frame, exiting")
                    break
                continue
            
            current_tracked_objects = {}
            
            movement_detected = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_contour_area: 
                    continue
                    
                (x, y, w, h) = cv2.boundingRect(contour)
                
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    if debug_mode:
                        cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (128, 0, 128), 1)
                    continue
                
                center_x = x + w // 2
                center_y = y + h // 2
                

                is_head = False
                is_rapid = False
                movement_speed = 0
                

                if 0.7 <= aspect_ratio <= 1.3 and center_y < frame_height * 0.4 and area < 15000:
                    is_head = True
                
                if tracked_objects:
                    closest_id = None
                    min_dist = float('inf')
                    
                    for obj_id, obj_data in tracked_objects.items():
                        prev_center = obj_data['center']
                        dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                        if dist < min_dist and dist < max_match_distance:
                            min_dist = dist
                            closest_id = obj_id
                    
                    if closest_id is not None:
                        prev_center = tracked_objects[closest_id]['center']
                        dx = center_x - prev_center[0]
                        dy = center_y - prev_center[1]
                        movement_speed = np.sqrt(dx**2 + dy**2)
                        
                        if movement_speed > rapid_movement_threshold:
                            is_rapid = True
                
                box_color = ARM_COLOR
                label = "Arm"
                
                if is_rapid:
                    
                    box_color = RAPID_COLOR
                    label = f"RAPID ({int(movement_speed)}px)"
                elif is_head:
                   
                    box_color = HEAD_COLOR
                    label = "Head"
                
               
                cv2.rectangle(frame_vis, (x, y), (x+w, y+h), box_color, 2)
                if debug_mode:
                    cv2.putText(frame_vis, label, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                
                movement_detected = True
                
                
                object_id = None
                
                if tracked_objects:
                    
                    closest_id = None
                    min_dist = float('inf')
                    
                    for obj_id, obj_data in tracked_objects.items():
                        prev_center = obj_data['center']
                        dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                        if dist < min_dist and dist < max_match_distance:
                            min_dist = dist
                            closest_id = obj_id
                    
                    if closest_id is not None:
                        
                        object_id = closest_id
                        prev_center = tracked_objects[object_id]['center']
                        
                        
                        dx = center_x - prev_center[0]
                        dy = center_y - prev_center[1]
                        
                        
                        magnitude = np.sqrt(dx**2 + dy**2)
                        if magnitude > min_movement_threshold:
                            
                            if object_id not in history_buffer:
                                history_buffer[object_id] = []
                            
                            
                            history_buffer[object_id].append((dx, dy))
                            
                            
                            if len(history_buffer[object_id]) > arrow_smoothing:
                                history_buffer[object_id].pop(0)
                            
                            
                            avg_dx = sum(item[0] for item in history_buffer[object_id]) / len(history_buffer[object_id])
                            avg_dy = sum(item[1] for item in history_buffer[object_id]) / len(history_buffer[object_id])
                            
                            
                            avg_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
                            if avg_magnitude > 0:
                                norm_dx = avg_dx / avg_magnitude
                                norm_dy = avg_dy / avg_magnitude
                                
                                
                                scaled_dx = int(norm_dx * arrow_scale)
                                scaled_dy = int(norm_dy * arrow_scale)
                                
                                
                                arrow_start = (center_x, center_y)
                                arrow_end = (center_x + scaled_dx, center_y + scaled_dy)
                                
                                
                                try:
                                    cv2.arrowedLine(frame_vis, 
                                                  arrow_start,
                                                  arrow_end,
                                                  ARROW_COLOR,
                                                  3,
                                                  tipLength=0.3)
                                    
                                    
                                    if debug_mode and not is_rapid:
                                        speed = int(magnitude)
                                        cv2.putText(frame_vis, f"{speed}px", 
                                                  (center_x + scaled_dx + 5, center_y + scaled_dy), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, ARROW_COLOR, 1)
                                except Exception as e:
                                    print(f"Error drawing arrow: {str(e)}")
                    else:
                        
                        object_id = next_id
                        next_id += 1
                else:
                    
                    object_id = next_id
                    next_id += 1
                
                
                if object_id is not None:
                    current_tracked_objects[object_id] = {
                        'center': (center_x, center_y),
                        'last_seen': time.time(),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'is_head': is_head,
                        'is_rapid': is_rapid,
                        'speed': movement_speed
                    }
                    
                   
                    if debug_mode:
                        cv2.putText(frame_vis, f"{object_id}", 
                                   (center_x, center_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            active_ids = set(current_tracked_objects.keys())
            history_ids = set(history_buffer.keys())
            for old_id in history_ids - active_ids:
                del history_buffer[old_id]
            
            tracked_objects = current_tracked_objects
            
            cv2.putText(frame_vis, f"FPS: {fps}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame_vis, f"Objects: {len(tracked_objects)}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if movement_detected:
                motion_count += 1
                cv2.putText(frame_vis, "Motion Detected", 
                           (frame_width - 170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Human Body Tracking', frame_vis)
            
            frame1 = frame2
            ret, frame2 = cap.read()
            if not ret:
                print("Error reading frame, exiting")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("---------------------")
    print("  Human Body Tracking")
    print("---------------------")
    start_motion_detection()
