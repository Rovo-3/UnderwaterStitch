import math

class PID():
    def __init__(self, Kp, Ki, Kd, Tf=1):
        self.set_gain(Kp,Ki,Kd, Tf)
        self.set_limit(None,None,0)
        self.interval = 1
        self.integral = 0
        self.last_error = 0
        # default for thruster
        self.upper = 2000
        self.lower = 1000
        self.midval = 1500
        self.x=0

    def set_gain(self, kp, ki, kd, tf):
        # update every gain
        self.Kp, self.Ki, self.Kd, self.Tf= kp, ki, kd, tf

    def set_limit(self, upper, lower, midval):
        self.upper, self.lower, self.midval= upper, lower, midval
        if self.upper == None:
            self.upper=float("inf")
        if self.lower == None:
            self.lower=-float("inf")

    def calculate(self, error, interval ,aw=False) -> int:

        """
        Calculating the PID, returning the output
        """
        if interval==-1:
            print("invalid interval, returning netral")
            return self.midval
        
        self.interval = interval

        P = self.Kp * error
        
        I = self.Ki * self.integral
        # filtering D
        if self.Kd != 0.0 and self.Tf > 0.0:
            Kn = 1.0 / self.Tf
            self.x = math.exp(-Kn * self.interval) * self.x - Kn * (1.0 - math.exp(-Kn * self.interval)) * self.Kd * error
            D = self.x + Kn * self.Kd * error
        else:
            # Simple derivative without filtering (fallback)
            D=self.Kd * (error - self.last_error) / self.interval if self.interval != 0 else 0
        print (P,I,D)
        output = P+I+D
        output += self.midval
        # update, error, integral
        self.last_error = error
        self.integral += error * self.interval

        if aw:
            if output>self.upper or output<self.lower:
                self.integral -= error * self.interval
            output= max(min(output,self.upper),self.lower)
        
        return int(output)
