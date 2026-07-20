from plutocontrol import Pluto
import time

MAX_LIMIT = 1600
MIN_LIMIT = 1400

def init_drone(my_pluto):
    # my_pluto = Pluto()
    my_pluto.connect()

    time.sleep(2)
    my_pluto.calibrate_acceleration()
    time.sleep(5)

    battery = my_pluto.get_battery()
    # battery_level = my_pluto.get_battery_level()
    print(f"Battery: {battery}")

    my_pluto.arm()
    time.sleep(2)

    smart_take_off(my_pluto)

    my_pluto.devOn()

    return my_pluto


def stabilise_drone(my_pluto, roll, pitch, yaw, throttle):
        
        # roll, pitch, yaw, throttle = max(MIN_LIMIT, min(MAX_LIMIT, roll)), max(MIN_LIMIT, min(MAX_LIMIT, pitch)), max(MIN_LIMIT, min(MAX_LIMIT, yaw)), max(MIN_LIMIT, min(MAX_LIMIT, throttle))
        
        my_pluto.rcRoll = roll #1500 is neutral <1500 is left 
        my_pluto.rcPitch = pitch #1500 is neutral <1500 is back
        my_pluto.rcYaw = yaw #1500 is neutral <1500 is left
        my_pluto.rcThrottle = throttle #1500 is neutral <1500 is down


def smart_take_off(my_pluto, timeout=1):
    print("Taking off...")
    my_pluto.take_off()
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(0.1) 
        
    print("Takeoff timeout reached. Ready for commands.")


def smart_land(my_pluto, timeout=7):
    print("Initiating Auto-Land... please wait 7 seconds.")
    
    # Neutralize controls to prevent drift
    hold_position(my_pluto)
    
    my_pluto.land()
    
    start_time = time.time()
    # Give the drone 7 seconds to physically reach the ground
    while time.time() - start_time < timeout:
        time.sleep(0.1)
        
    print("Landing timeout reached. Ground assumed.")


def hold_position(my_pluto):
    my_pluto.rcRoll = 1500
    my_pluto.rcPitch = 1500
    my_pluto.rcYaw = 1500
    my_pluto.rcThrottle = 1500


def fail_safe(my_pluto):
        smart_land(my_pluto)
        time.sleep(1)
        my_pluto.disarm()
        time.sleep(1)
        my_pluto.disconnect()