import time


class SimplePID:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()

    def compute(self, current_value):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = 1e-4
            
        error = current_value - self.setpoint
        self.integral += error * dt
        
        # Anti-windup cap
        self.integral = max(-100, min(100, self.integral))
        
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        self.last_time = now
        return output