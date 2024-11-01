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
veloPID = True
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
def getData(file_name,type="sensor",retries=5,delay=0.1):
    if type == "sensor":
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
    if type == "control":
        for attempt in range(retries):
            try:
                # Attempt to open and load the file
                with open(file_name, "r") as target_file:
                    data = json.load(target_file)
                print("Data successfully loaded.")
                return data["Forward"], data["Yaw"], data["Depth"]
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
    wp_heading,length,gap,iteration = 90,10,2,2
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
    control_file_name = file_directory+"/PID_parameter.json"
    forward_param, yaw_param, depth_param = getData(control_file_name, "control")
    yaw_control = PID(Kp=yaw_param["kp"],Ki=yaw_param["ki"],Kd=yaw_param["kd"],Tf=0)

    # 5. Determine the maximum, minimum, and midval of control signal
    yaw_control.set_limit(upper=1650,lower=1350, midval=1500)
    
    target_depth=-1
    depth_control = PID(Kp=depth_param["kp"],Ki=depth_param["ki"],Kd=depth_param["kd"],Tf=0)
    depth_control.set_limit(upper=1650,lower=1350, midval=1500)

    vehicle_position = np.array([init_pos])
    yaw_data_last, depth_data_last = 0 , 0
    # 6. Start the loop for control mechanism
    while True:
        # Check the sensor data for PID input
        dvl_data, depth = getData(file_name)
        # Calculate the Guidance
        curr_heading=dvl_data["yaw"]
        curr_heading = (curr_heading+math.pi)%(2*math.pi)-math.pi
        curr_depth = depth
        curr_pos = (dvl_data["x"],dvl_data["y"])
        target_angle, target_point, status = guidance.calculate_steering(vehicle_position=curr_pos,vehicle_heading=curr_heading, velocity=1)
        
        # Calculate the error from sensor data and Guidance
        heading_error = target_angle-curr_heading
        heading_error = normalize_yaw(heading_error)
        depth_error = target_depth-curr_depth

        # velocity PID 
        # ***comment if you want regular PID 
        yaw_velo_des = pos_error_to_velo(heading_error, 1, 1, 0.1)
        depth_velo_des = pos_error_to_velo(depth_error, 1, 2, 0.1)
        # ***comment if you want regular PID 
        
        
        print("Curr, WP, Guidance, Error, Heading, Target: ",curr_heading,wp_heading,target_angle,heading_error, target_point)
        # Calculate the PID output
        time_now=time.time()
        global time_last
        interval = time_now-time_last
        if (interval>5):
            interval=-1

        # velocity PID 
        # ***comment if you want regular PID 
        if veloPID:
            depth_velo_now = calc_velo(curr_depth, depth_data_last, interval)
            depth_velo_error = depth_velo_des - depth_velo_now
            yaw_velo_now = calc_velo(curr_heading, yaw_data_last, interval)
            yaw_velo_error = yaw_velo_des - yaw_velo_now
            output_depth = depth_control.calculate(depth_velo_error,interval,True)
            output_yaw = yaw_control.calculate(yaw_velo_error,interval,True)
        # ***comment if you want regular PID 

        if not veloPID:
            output_yaw = yaw_control.calculate(heading_error,interval=interval,aw=True)
            output_depth = depth_control.calculate(depth_error, interval=interval, aw=True)
        
        # updating variables
        time_last=time_now
        yaw_data_last, depth_data_last = curr_heading ,curr_depth 

        # Input the output value to ROV
        # print(output_yaw)

        # yawing
        # manual_control(*control_mapping[parameter])
        set_rc(4, int(output_yaw))

        # forward
        set_rc(5, 1530)

        # dive
        set_rc(3,output_depth)
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

def tuning_PID():
    print("Starting the PID Tuning")
    print("Do not tune in air!")
    is_tuning_yaw = False
    is_tuning_depth = False
    control_file_name = file_directory+"/PID_parameter.json"
    forward_param, yaw_param, depth_param = getData(control_file_name, "control")
    mode = input("yaw or depth?")
    param=None
    if mode == "yaw":
        is_tuning_yaw = True
        yaw_desired = float(input("designated_yaw (angle)? "))
        yaw_rad = np.deg2rad(yaw_desired)
        yaw_desired = np.arctan2(np.sin(yaw_rad),np.cos(yaw_rad))
        print(yaw_desired)
        param = yaw_param
    elif mode == "depth":
        is_tuning_depth=True
        depth_desired = -float(input("desired_depth? "))
        param = depth_param
    kp,ki,kd = param["kp"], param["ki"], param["kd"]
    pid_control = PID(kp,ki,kd,0)
    pid_control.set_limit(upper=1650,lower=1350,midval=1500)
    # loop time
    loop_time = 100
    data_to_visualized = np.array([0,0,0,0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("P:{}, I:{}, D:{}".format(kp,ki,kd))
    data_last = 0
    # Main Loop
    for i in range(loop_time):
        sensor_file_name = file_directory+"/sensor_data.json"
        dvl_data, depth_current = getData(sensor_file_name)
        yaw_current = normalize_yaw(dvl_data["yaw"])
        # yaw_current = (yaw_current+math.pi)%(2*math.pi)-math.pi
        print("current yaw",yaw_current)
        if is_tuning_yaw:
            error = yaw_desired-yaw_current
            error = normalize_yaw(error)
            velo_des = pos_error_to_velo(error, 1, 0.5, 0.1)
            channel = 4
            data = yaw_current
        elif is_tuning_depth:
            error = (depth_desired-depth_current)
            velo_des = pos_error_to_velo(error,1, 2, 0.05)
            data = depth_current
            channel = 3
        print(error)
        time_now=time.time()
        global time_last
        interval = time_now-time_last
        if (interval>5):
            interval=-1
        print("Interval, ", interval)

        output = int(pid_control.calculate(error,interval,True))
        if veloPID:
            velo_now = calc_velo(data, data_last, interval)
            velo_error = velo_des - velo_now
            output_velo_control = int(pid_control.calculate(velo_error,interval,True))
        set_rc(channel,output_velo_control)

        data_to_visualized = np.vstack([data_to_visualized, np.array([i,data,error,output])])
        visualization(axs[0],data_to_visualized[:, 0],data_to_visualized[:, 1], "r-", "{}".format(mode).capitalize(), f"Vehicle {mode} Data")
        visualization(axs[1],data_to_visualized[:, 0],data_to_visualized[:, 2], "b-", "Error{}".format(mode), f"Error {mode} Data")
        visualization(axs[2],data_to_visualized[:, 0],data_to_visualized[:, 3], "g-", "Output", "Output Thruster Value")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        time_last=time_now
        data_last = data
        time.sleep(0.5)
        # plt.figure(1)
        # plt.plot(data_to_visualized[:, 0], data_to_visualized[:, 1], 'r-', label=mode)
        # plt.title("Figure 3: Output")
        # plt.xlabel("Time")
        # plt.ylabel("Output Value")
        # plt.legend()
        # plt.draw()
        # plt.pause(0.01)
        # time.sleep(0.5)

        # plt.figure(2)
        # plt.plot(data_to_visualized[:, 0], data_to_visualized[:, 2], 'b-', label="Error{}".format(mode))
        # plt.draw()
        # plt.pause(0.01)
        # time.sleep(0.5)

        # plt.figure(3)
        # plt.plot(data_to_visualized[:, 0], data_to_visualized[:, 3], 'g-', label="Output")
        # plt.draw()
        # plt.pause(0.01)
        # time.sleep(0.5)
    file_path = "{}/../../output/tuning_{}_kp{}_ki{}_kd{}.jpg".format(file_directory,mode,kp,ki,kd)
    print("Saving to:", file_path)
    plt.savefig(file_path)

def visualization(axis, data_x, data_y,linesetting,label, title):
    axis.cla()
    axis.plot(data_x, data_y, linesetting, label=label)
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.legend()

def normalize_yaw(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def pos_error_to_velo(error_now, max_velo, max_error, min_error):
    if error_now != 0:
        sign = abs(error_now)/error_now
    else:
        sign = 1
    if (abs(error_now)>max_error):
        velo_desired = max_velo * sign
    elif (abs(error_now)<min_error):
        velo_desired =  0
    elif (abs(error_now)>min_error):
        velo_desired = max_velo * (error_now-min_error)/max_error-min_error
    print("velo_des",velo_desired)
    return velo_desired

def calc_velo(pos_now, pos_last, interval):
    if interval!=-1:
        velo_now = (pos_now-pos_last)/interval
    else: 
        velo_now=0
    return velo_now
        
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
    TUNING = "tuning_pid"

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
                        10. tuning_pid
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
        if command == Commands.TUNING.value:
            tuning_PID()
        # else:
        #     print("Unkown Command, ", command)
        
    except KeyboardInterrupt:
        print("Bye")
        break