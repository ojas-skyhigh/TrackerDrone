import cv2
from vision.object_detection import *
from flightcontroller.drone_control import *
from flightcontroller.PID_control import *  
from plutocontrol import Pluto
import time

my_pluto = Pluto()

pid_roll = SimplePID(kp=1.0, ki=0.0, kd=0.0, setpoint=320)   # Screen X center
pid_throttle = SimplePID(kp=1.0, ki=0.0, kd=0.0, setpoint=240)  # Screen Y center
pid_pitch = SimplePID(kp=40.0, ki=0.0, kd=0.0, setpoint=0.9) # Real distance Z target

def move_drone(x, y, z):
    roll_out = pid_roll.compute(x)
    throttle_out = pid_throttle.compute(y)
    pitch_out = pid_pitch.compute(z)

    # THE FIX: Ground-Camera Coordinate Mapping
    raw_roll = 1500 - roll_out      # If X is < 320 (Left), this outputs > 1500 (Move Right)
    raw_throttle = 1700 + throttle_out # If Y is < 240 (Too High), this outputs < 1500 (Move Down)
    raw_pitch = 1500 - pitch_out    # If Z is > 0.9m (Too Far), this outputs < 1500 (Move Backward)
    yaw = 1500  

    roll = max(1400, min(1600, int(raw_roll)))
    pitch = max(1400, min(1600, int(raw_pitch)))
    throttle = max(1600, min(1800, int(raw_throttle)))

    print(f"Roll: {roll}, Pitch: {pitch}, Throttle: {throttle}")
    
    stabilise_drone(my_pluto, roll, pitch, yaw, throttle)
    

def main():
    global my_pluto

    try:
        print("Initializing tracking system...")
        model, pipeline, align = init_tracker()
        coordinate_stream = stream_obj_coords(model, pipeline, align) 
        
        print("System active.")
        print("Waiting for target lock... (Press 'w' in the video window to quit)")

        for coordinates in coordinate_stream:
            if cv2.waitKey(1) & 0xFF == ord('w'):
                print("Abort commanded before takeoff.")
                return

            if coordinates is not None:
                print("Drone locked in frame! Initiating takeoff sequence...")
                break 
        
        print("Initializing drone...")
        my_pluto = init_drone(my_pluto)
        print("Drone hovering. Tracking engaged. (Press 'q' to emergency land)")
        
        for coordinates in coordinate_stream:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[EMERGENCY] 'q' pressed. Triggering fail-safe...")
                fail_safe(my_pluto)
                break
            
            if coordinates is None:
                print("TRACKING LOST - Neutralizing controls") 
                stabilise_drone(my_pluto, 1500, 1500, 1700, 1500)
                continue
                
            x, y, z = coordinates
            print(f"Target Coordinates -> Screen X: {x}px, Screen Y: {y}px, Distance Z: {z:.2f}m")
            move_drone(x, y, z)
            
    except KeyboardInterrupt:
        print("\n[WARNING] Keyboard interrupt received. Triggering fail-safe...")
        fail_safe(my_pluto)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        fail_safe(my_pluto)

if __name__ == "__main__":
    main()