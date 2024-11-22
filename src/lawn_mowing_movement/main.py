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
from threading import Thread

# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14770')

# Wait a heartbeat before sending commands
master.wait_heartbeat()
print("get heartbeat")
file_directory = os.path.dirname(os.path.abspath(__file__))
time_last=0
veloPID = False
simulation = False

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

# checking message
def check_msg(msg_type=None):
    print("Checking Message")
    if msg_type == None:
        # just mention the power status with blocking true
        msg = master.recv_match(type="POWER_STATUS", blocking=True)
        msg = msg.to_dict()
        print("Got message: ", msg)

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
    for attempt in range(retries):
        try:
            # Attempt to open and load the file
            with open(file_name, "r") as target_file:
                data = json.load(target_file)
            print("Data successfully loaded.")
            # if sensor
            if type=="sensor":
                return data["DVL"], data["depth"]
            # if control
            elif type=="control":
                return data["Forward"], data["Yaw"], data["Depth"], data["Setpoint"]
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

    wp_heading,length,gap,iteration = 270,5,5,1
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
    lookahead_distance = 0.2
    guidance = Guidance(waypoints=waypoints, mode="Stanley", lookahead_distance=lookahead_distance,k_e=0.5, distance_treshold=0.2, generate_virtual_wp=False)

    # 4. Instantiate PID yaw
    control_file_name = file_directory+"/PID_parameter.json"
    forward_param, yaw_param, depth_param, _ = getData(control_file_name, "control")
    yaw_control = PID(Kp=yaw_param["kp"],Ki=yaw_param["ki"],Kd=yaw_param["kd"])

    # 5. Determine the maximum, minimum, and midval of control signal
    yaw_control.set_limit(upper=1750,lower=1250, midval=1500)
    
    target_depth = -0.2
    depth_control = PID(Kp=depth_param["kp"],Ki=depth_param["ki"],Kd=depth_param["kd"])
    depth_control.set_limit(upper=1700,lower=1300, midval=1500)

    forward_control = PID(Kp=forward_param["kp"],Ki=forward_param["ki"],Kd=forward_param["kd"])
    forward_control.set_limit(upper=1700, lower=1300, midval=1500)

    lateral_control = PID(Kp=forward_param["kp"],Ki=forward_param["ki"],Kd=forward_param["kd"])
    lateral_control.set_limit(upper=1700, lower=1300, midval=1500)

    vehicle_position = np.array([init_pos])

    # PID velo
    yaw_data_last, depth_data_last = 0, 0

    print("Starting the Mission, arming")
    arm()
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    # 6. Start the loop for control mechanism
    while True:
        # Check the sensor data for PID input
        dvl_data, depth = getData(file_name)
        # Calculate the Guidance
        curr_heading = dvl_data["yaw"]
        curr_heading = (curr_heading+math.pi)%(2*math.pi)-math.pi
        curr_depth = depth
        curr_pos = (dvl_data["x"],dvl_data["y"])
        target_angle, target_point, status, forward_velo, lateral_velo = guidance.calculate_steering(vehicle_position=curr_pos,vehicle_heading=curr_heading, velocity=1)
        
        # Calculate the error from sensor data and Guidance
        heading_error = target_angle-curr_heading
        heading_error = normalize_yaw(heading_error)
        depth_error = target_depth-curr_depth
        forward_error = forward_velo-(dvl_data["vx"]**2+dvl_data["vy"]**2)**0.5*math.cos(target_angle)
        lateral_error = lateral_velo-(dvl_data["vx"]**2+dvl_data["vy"]**2)**0.5*math.sin(target_angle)

        print("Curr, WP, Guidance, Error, Heading, Target: ",curr_heading,wp_heading,target_angle,heading_error, target_point)
        # Calculate the PID output
        time_now=time.time()
        global time_last
        interval = time_now-time_last
        if (interval>5):
            interval=-1
        
        # # PID velo 
        # ***comment if you want regular PID 
        yaw_velo_des = pos_error_to_velo(heading_error, 1, 0.5, 0.01)
        depth_velo_des = pos_error_to_velo(depth_error, 1, 1.5, 0.05)
        # ***comment if you want regular PID 

        # PID velo
        if veloPID:
            depth_velo_now = calc_velo(curr_depth, depth_data_last, interval)
            depth_velo_error = depth_velo_des - depth_velo_now
            yaw_velo_now = calc_velo(curr_heading, yaw_data_last, interval)
            yaw_velo_error = yaw_velo_des - yaw_velo_now
            output_depth = depth_control.calculate(depth_velo_error,interval,True)
            output_yaw = yaw_control.calculate(yaw_velo_error,interval,True)
            yaw_data_last, depth_data_last = curr_heading ,curr_depth

        if not veloPID:
            output_yaw = yaw_control.calculate(heading_error,interval=interval,aw=True)
            output_depth = depth_control.calculate(depth_error, interval=interval, aw=True)
            output_forward = forward_control.calculate(forward_error, interval=interval, aw=True)
            output_lateral = forward_control.calculate(lateral_error, interval=interval, aw=True)
        
        # updating variables
        time_last=time_now
         

        # yawing
        set_rc(4, int(output_yaw))
        # forward
        set_rc(5, int(output_forward))
        # dive
        set_rc(3,output_depth)

        if simulation == True:
            output_value_diff = int(output_yaw)-1500
            if (abs(output_value_diff) <30) and (output_value_diff != 0):
                sign = (output_yaw-1500)/abs(output_yaw-1500)
                set_rc(4, int(1500+sign*30))
                set_rc(4, 1500)

        heading_error_deg = math.degrees(heading_error)
        print("Heading Error: ", heading_error_deg)

        print(status)
        # Repeat till guidance said enough
        if(status=="Guidance Done"):
            # break from the loop
            set_rc(4, 1500)
            set_rc(5, 1500)
            print("End of mission, disarming")
            disarm()
            break
        
        # visualization
        vehicle_position = np.vstack([vehicle_position,np.array(curr_pos)])
        plt.clf()
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', label='Waypoints')
        plt.plot(vehicle_position[:, 0], vehicle_position[:, 1], 'bo-', label='Vehicle Position')
        plt.title("Lawn Mowing Result")
        plt.legend()
        plt.draw()
        plt.pause(0.01)
        time.sleep(0.2)

        file_path = "{}/../../output/{}guidance_test.jpg".format(file_directory, time_str)
        print("Saving to:", file_path)
        plt.savefig(file_path)

def tuning_PID():
    print("Starting the PID Tuning")
    print("Do not tune in air!")
    is_tuning_yaw = False
    is_tuning_depth = False
    control_file_name = file_directory+"/PID_parameter.json"
    forward_param, yaw_param, depth_param, setpoint = getData(control_file_name, "control")
    mode = input("yaw or depth?")
    param=None
    if mode == "yaw":
        is_tuning_yaw = True
        # yaw_desired = float(input("designated_yaw (angle)? "))
        yaw_rad = np.deg2rad(setpoint["yaw"])
        yaw_desired = np.arctan2(np.sin(yaw_rad),np.cos(yaw_rad))
        data_desired = yaw_desired
        print(yaw_desired)
        param = yaw_param
    else: #mode == "depth":
        mode = "depth"
        is_tuning_depth=True
        # depth_desired = -float(input("desired_depth? "))
        depth_desired = -setpoint["depth"]
        data_desired=depth_desired
        param = depth_param
    kp,ki,kd = param["kp"], param["ki"], param["kd"]
    pid_control = PID(kp,ki,kd,0)
    pid_control.set_limit(upper=1600,lower=1400,midval=1500)
    # loop time
    loop_time = 1000
    data_to_visualized = np.array([0,0,0,0,0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("P:{}, I:{}, D:{}".format(kp,ki,kd))
    data_last = 0
    arm()
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    # Main Loop
    for i in range(loop_time):
        sensor_file_name = file_directory+"/sensor_data.json"
        dvl_data, depth_current = getData(sensor_file_name)
        _, param_yaw, param_depth, setpoint = getData(control_file_name, "control")
        data_desired = setpoint[mode]
        if mode == "yaw":
            yaw_rad = np.deg2rad(setpoint["yaw"])
            param=param_yaw
            data_desired = np.arctan2(np.sin(yaw_rad),np.cos(yaw_rad))
        else:
            data_desired *= -1
            param = param_depth
        kp,ki,kd = param["kp"], param["ki"], param["kd"]
        pid_control.set_gain(kp,ki,kd,0)
        # this is not necessary for actual dvl data
        yaw_current = normalize_yaw(dvl_data["yaw"])
        # yaw_current = (yaw_current+math.pi)%(2*math.pi)-math.pi
        print("current yaw",yaw_current)
        if is_tuning_yaw:
            error = data_desired-yaw_current
            error = normalize_yaw(error)
            velo_des = pos_error_to_velo(error, 0.5, 1, 0)
            channel = 4
            data = yaw_current
        elif is_tuning_depth:
            error = (data_desired-depth_current)
            velo_des = pos_error_to_velo(error,1, 1.5, 0.05)
            data = depth_current
            channel = 3
        print(error)
        # calculating control interval
        time_now=time.time()
        global time_last
        interval = time_now-time_last
        if (interval>5):
            interval=-1
        print("Interval, ", interval)

        output = int(pid_control.calculate(error,interval,True))
        # recalculated if veloPID
        if veloPID:
            velo_now = calc_velo(data, data_last, interval)
            velo_error = velo_des - velo_now
            output = int(pid_control.calculate(velo_error,interval,True))
        set_rc(channel,output)

        data_to_visualized = np.vstack([data_to_visualized, np.array([i,data,data_desired,error,output,])])
        visualization(axs[0],data_to_visualized[:, 0],data_to_visualized[:, 1], "k-", "{}".format(mode).capitalize(), f"Vehicle {mode} Data", 1)
        visualization(axs[0],data_to_visualized[:, 0],data_to_visualized[:, 2], "r-", "Setpoint {}".format(mode).capitalize(), f"Vehicle {mode} Data", 0)
        visualization(axs[1],data_to_visualized[:, 0],data_to_visualized[:, 3], "b-", "Error{}".format(mode), f"Error {mode} Data")
        visualization(axs[2],data_to_visualized[:, 0],data_to_visualized[:, 4], "g-", "Output", "Output Thruster Value")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        time_last=time_now
        data_last = data
        time.sleep(0.2)

        # Saving every iteration
        file_path = "{}/../../output/{}tuning_{}_kp{}_ki{}_kd{}_{}.jpg".format(file_directory,time_str,mode,kp,ki,kd,veloPID)
        print("Saving to:", file_path)
        plt.savefig(file_path)

    set_rc(3,1500)
    set_rc(4,1500)
    disarm()

def visualization(axis, data_x, data_y,linesetting,label, title, clear=1):
    if clear == 1:
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
        
        print("Okay, ", command)
        if command == Commands.ARM.value:
            arm()
        if command == Commands.DISARM.value:
            disarm()
        if command == Commands.CHECK_DATA.value:
            file_name = file_directory+"/sensor_data.json"
            getData(file_name=file_name)
        if command == Commands.YAW_RIGHT.value:
            arm()
            set_rc(4,1526)
            set_rc(4,1500)
        if command == Commands.YAW_LEFT.value:
            arm()
            set_rc(4,1474)
        if command.startswith(Commands.MANUAL_CONTROL.value):
            _,parameter = command.split()
            control_mapping = {
                "forward": (300, 0, 500, 0),
                "backward": (-300, 0, 500, 0),
                "strive_right": (0, 300, 500, 0),
                "strive_left": (0, -300, 500, 0),
                "up": (0, 0, 700, 0),
                "down": (0, 0, 300, 0),
                "turn_left": (0, 0, 500, 1000),
                "turn_right": (0, 0, 500, -1000)
            }
            if parameter in control_mapping:
                print(parameter)
                # passing (unpack) dict values in tuples to the arguments of manual_control
                for i in range(100):
                    manual_control(*control_mapping[parameter])
            else:
                print("Unknown manual control parameter")
        if command == Commands.LM_MOVE.value:
            start_LM_mission()
        if command == Commands.STOP.value:
            arm()
            set_rc(4,1500)
            print("Stopping the thruster")
            disarm()
        if command == Commands.TUNING.value:
            tuning_PID()
        else:
            print("Unkown Command, ", command)
        
    except KeyboardInterrupt:
        set_rc(4,1500)
        set_rc(3,1500)
        disarm()
        print("Bye")
        break