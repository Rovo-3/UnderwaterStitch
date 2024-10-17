# Import mavutil
from pymavlink import mavutil
from enum import Enum
import time
from guidance import Guidance
from control import PID
# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14770')
# Wait a heartbeat before sending commands
# master.wait_heartbeat()
print("get heartbeat")

# https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM

def arm() :
    master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 0, 0, 0, 0, 0, 0)
    print("Arming")

def disarm() :
    master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 0, 0, 0, 0, 0, 0, 0)
    print("Disarming")

def check_msg():
    print("Checking Message")
    msg = master.recv_match(type="POWER_STATUS", blocking=True)
    msg = msg.to_dict()
    print("Got message: ", msg)
    # if not msg:
    #     return
    # if msg.get_type() == 'HEARTBEAT':
    #     print("\n\n**Got message: %s**" % msg.get_type())
    #     print("Message: %s" % msg)
    #     print("\nAs dictionary: %s" % msg.to_dict())
    #     # Armed = MAV_STATE_STANDBY (4), Disarmed = MAV_STATE_ACTIVE (3)
    #     print("\nSystem status: %s" % msg.system_status)

def set_rc(channel_id,pwm=1500):
    print(f"RC channel {channel_id} set to PWM {pwm}.")
    if channel_id < 1 or channel_id > 18:
        print("Channel does not exist.")
        return
    # Mavlink 2 supports up to 18 channels:
    # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
    rc_channel_values = [65535 for _ in range(18)]
    rc_channel_values[channel_id - 1] = pwm
    master.mav.rc_channels_override_send(
        master.target_system,                # target_system
        master.target_component,             # target_component
        *rc_channel_values)                  # RC channel list, in microseconds.
    
def manual_control(x=0,y=0,z=500,r=0):
    for _ in range (100):
        master.mav.manual_control_send(
        master.target_system,
        x,
        y,
        z,
        r,
        0)

def start_LM_mission():
    print("Starting the LM Mission")
    # 1. Input for degree, length and gap.
    # 2. Generate WP
    # 3. Instantiate Guidance
    # 4. Instantiate PID yaw and forward speed control
    # 5. Determine the maximum and minimum control signal
    # 6. Start the loop for control mechanism
        # Check the sensor data for PID input
        # Calculate the Guidance
        # Calculate the error from sensor data and Guidance
        # Calculate the PID output
        # Input the output value to ROV
        # Repeat till guidance said enough

class Commands(Enum):
    DISARM = "disarm"
    ARM = "arm"
    MANUAL_CONTROL = "manual_control"
    CHECK_MSG = "check_msg"
    YAW_RIGHT = "yaw_right"
    YAW_LEFT = "yaw_left"
    STOP = "stop"
    CONNECTION = "connection"
    LM_MOVE = "LM_move"

while True:
    try:
        command = input("""Put Command:
                        1. disarm
                        2. arm
                        3. manual_control param
                        4. check_msg
                        5. connection
                        6. yaw_right
                        7. yaw_left
                        8. LM_move
                        9. stop
                        """).strip()
        
        # split_command = command.split()
        # if len(split_command) > 1:
        #     command, parameter = split_command
        #     print(parameter)
        print("Okay, ", command)
        if command == Commands.ARM.value:
            arm()
        if command == Commands.DISARM.value:
            disarm()
        if command == Commands.ARM.value:
            check_msg()
        if command == Commands.YAW_RIGHT.value:
            set_rc(4,1530)
        if command == Commands.YAW_LEFT.value:
            set_rc(4,1470)
        if command.startswith(Commands.MANUAL_CONTROL.value):
            _,parameter = command.split()
            control_mapping = {
                "forward": (300, 0, 500, 0),
                "backward": (-300, 0, 500, 0),
                "strive_right": (0, 300, 500, 0),
                "strive_left": (0, -300, 500, 0),
                "up": (0, 0, 700, 0),
                "down": (0, 0, 300, 0),
                "turn_left": (0, 0, 500, 300),
                "turn_right": (0, 0, 500, -300)
            }
            if parameter in control_mapping:
                # passing (unpack) dict values in tuples to the arguments of manual_control
                manual_control(*control_mapping[parameter])
            else:
                print("Unknown manual control parameter")
        if command == Commands.LM_MOVE.value:
            start_LM_mission()
        if command == Commands.STOP.value:
            set_rc(4,1500)
            print("Stopping the thruster")
        # else:
        #     print("Unkown Command, ", command)
        
    except KeyboardInterrupt:
        print("Bye")
        break