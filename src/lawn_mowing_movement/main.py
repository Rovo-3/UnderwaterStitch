# Import mavutil
from pymavlink import mavutil
from enum import Enum
import time
from guidance import Guidance
from control import PID
from waypoint_generator import SweepWPGenerator
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14770')
# Wait a heartbeat before sending commands
master.wait_heartbeat()
print("get heartbeat")
file_directory = os.path.dirname(os.path.abspath(__file__))
time_last=0
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

def check_msg(msg_type=None):
    print("Checking Message")
    if msg_type == None:
        # just mention the power status with blocking true
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
def getData(file_name,retries=5,delay=0.1):
    for attempt in range(retries):
        try:
            # Attempt to open and load the file
            with open(file_name, "r") as target_file:
                data = json.load(target_file)
            print("Data successfully loaded.")
            return data["DVL"], data["depth"]
        except (FileNotFoundError, PermissionError) as e:
            print(f"Attempt {attempt + 1}: File not accessible - {e}. Retrying...")
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: Failed to decode JSON - {e}. Retrying...")
        # Wait before retrying
        time.sleep(delay)
    # Raise an exception if retries are exhausted
    raise Exception(f"Failed to read the file after {retries} attempts.")
    
    
def start_LM_mission():
    print("Starting the LM Mission")
    # 1. Input for degree, length and gap.
    # heading = math.radians(int(input("Input the heading in degree ")))
    # length = int(input("Input the length of survey "))
    # gap = int(input("Input the gap of each mowing "))
    # iteration = int(input("How many iteration? "))
    wp_heading,length,gap,iteration = 200,10,2,2
    # regulate to [-pi,pi]
    wp_heading = (math.radians(wp_heading)+math.pi)%(2*math.pi)-math.pi

    # 2. Generate WP
    file_name = file_directory+"/sensor_data.json"
    dvl_data, depth = getData(file_name)
    # normalize the heading
    dvl_data["yaw"] = (dvl_data["yaw"]+math.pi)%(2*math.pi)-math.pi
    init_pos = (dvl_data["x"],dvl_data["y"])

    wp_generator = SweepWPGenerator(init_pos=init_pos, length=length, angle=wp_heading, gap=gap,iteration=iteration)
    waypoints = wp_generator.generate()
    waypoints = np.array(waypoints)
    
    # 3. Instantiate Guidance
    lookahead_distance = 0.1
    guidance = Guidance(waypoints=waypoints, mode="Stanley", lookahead_distance=lookahead_distance,k_e=1, distance_treshold=1, generate_virtual_wp=False)

    # 4. Instantiate PID yaw
    yaw_control = PID(Kp=100,Ki=0,Kd=0,Tf=1)

    # 5. Determine the maximum, minimum, and midval of control signal
    yaw_control.set_limit(upper=1700,lower=1300, midval=1500)

    vehicle_position = np.array([init_pos])

    # 6. Start the loop for control mechanism
    while True:
        # Check the sensor data for PID input
        dvl_data, depth = getData(file_name)
        # Calculate the Guidance
        curr_heading=dvl_data["yaw"]
        curr_heading = (curr_heading+math.pi)%(2*math.pi)-math.pi
        curr_pos = (dvl_data["x"],dvl_data["y"])
        target_angle, target_point, status = guidance.calculate_steering(vehicle_position=curr_pos,vehicle_heading=curr_heading)
        # Calculate the error from sensor data and Guidance
        heading_error = target_angle-curr_heading
        print("Curr, WP, Guidance, Error, Heading, Target: ",curr_heading,wp_heading,target_angle,heading_error, target_point)
        # Calculate the PID output
        time_now=time.time()
        global time_last
        interval = time_now-time_last
        if (time_now>2):
            interval=0
        output_yaw = yaw_control.calculate(heading_error,interval=interval,aw=True)
        time_last=time_now
        # Input the output value to ROV
        print(output_yaw)
        # yawing
        manual_control(*control_mapping[parameter])
        set_rc(4, int(output_yaw))
        # forward
        set_rc(5, 1530)
        print(status)
        # Repeat till guidance said enough
        if(status=="Guidance Done"):
            # break from the loop
            set_rc(4, 1500)
            # forward
            set_rc(5, 1500)
            break
        # visualization
        vehicle_position = np.vstack([vehicle_position,np.array(curr_pos)])
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', label='Waypoints')
        plt.plot(vehicle_position[:, 0], vehicle_position[:, 1], 'bo-', label='Waypoints')
        plt.draw()
        plt.pause(0.01)
        time.sleep(0.5)

class Commands(Enum):
    DISARM = "disarm"
    ARM = "arm"
    MANUAL_CONTROL = "manual_control"
    CHECK_DATA = "check_data"
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
                        4. check_data
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
        if command == Commands.CHECK_DATA.value:
            file_name = file_directory+"/sensor_data.json"
            getData(file_name=file_name)
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