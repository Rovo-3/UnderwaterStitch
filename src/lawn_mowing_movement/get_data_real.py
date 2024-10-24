# Open or create a log file in write mode
import time
import datetime
from pymavlink import mavutil
import json
import socket
import os

# Get the directory of the current Python file
file_directory = os.path.dirname(os.path.abspath(__file__))

print("Directory:", file_directory)
import argparse

class Sensors:
    def __init__(self, time_step=0.5):
        UdpIpPort = "udpin:0.0.0.0:14771"
        self.conn = mavutil.mavlink_connection(UdpIpPort)
        self.conn.wait_heartbeat()
        self.boot_time = time.time()
        self.time_step = time_step
        self.dvl_from_socket = False
        self.is_simulation = False
        # if the getting DVL data from socket
        if self.dvl_from_socket:
            self.dvlhost = '192.168.2.95'
            self.dvlport = 16171
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.dvlhost,self.dvlport))
            self.socket.setblocking(0)
        self.file_name = file_directory+"/sensor_data" + ".json"
        self.data_log = {
            "Date": 0,
            "Time": 0,
            "roll(degree)": 0,
            "pitch(degree)": 0,
            "yaw(degree)": 0,
            "rollspeed(w)": 0,
            "pitchspeed(w)": 0,
            "yawspeed(w)": 0,
            "airspeed(m/s)": 0,
            "groundspeed(m/s)": 0,
            "heading(degree)": 0,
            "depth": 0,
            "climbspeed(m/s)": 0,
            "xacc": 0,
            "yacc": 0,
            "zacc": 0,
            "altitude":0,
            "droll":0,
            "dpitch":0,
            "dyaw":0,
            "dx":0,
            "dy":0,
            "dz":0,
            "roll":0,
            "pitch":0,
            "yaw":0,
            "dvl_roll":0,
            "dvl_pitch":0,
            "dvl_yaw":0,
            "x":0,
            "y":0,
            "z":0,
            "vx":0,
            "vy":0,
            "vz":0,
            "alt":0,
            "lat":0,
            "lon":0,
            "velocity_valid":0,
            "covariance":0,
        }
        # self.createHeader()

    # def createHeader(self):
    #     keys = list(self.data_log.keys())
    #     header = ",".join(keys) + "\n"
    #     self.target_file.write(header)
    #     self.target_file.flush()

    def getData(self):
        time_now = self.getTime()
        date = self.getDate()
        # self.GetIMU(), self.GetVFRHUD(), self.GetAccell()
        if self.is_simulation:
            self.getPosition()
            self.getAttitude()
        # getDVLMav has yet tested
        self.getDVLSocket() if self.dvl_from_socket else self.getDVLMav()
        self.data_log["Date"] = date
        self.data_log["Time"] = time_now

    def getIMU(self):
        imuData = self.conn.recv_match(type="ATTITUDE", blocking=False)
        if imuData is not None:
            imu = imuData.to_dict()
            self.data_log["roll(degree)"] = imu["roll"] * 180 / 3.14
            self.data_log["pitch(degree)"] = imu["pitch"] * 180 / 3.14
            self.data_log["yaw(degree)"] = imu["yaw"] * 180 / 3.14
            self.data_log["rollspeed(w)"] = imu["rollspeed"] * 180 / 3.14
            self.data_log["pitchspeed(w)"] = imu["pitchspeed"] * 180 / 3.14
            self.data_log["yawspeed(w)"] = imu["yawspeed"] * 180 / 3.14

    def getPosition(self):
        position = self.conn.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if position is not None:
            position = position.to_dict()
            self.data_log["x"] = position["x"]
            self.data_log["y"] = position["y"]
            self.data_log["z"] = position["z"]
            self.data_log["vx"] = position["vx"]
            self.data_log["vy"] = position["vy"]
            self.data_log["vz"] = position["vz"]
            
    def getAttitude(self):
        attitude = self.conn.recv_match(type="AHRS2", blocking=False)
        if attitude is not None:
            attitude = attitude.to_dict()
            self.data_log["dvl_roll"] = attitude["roll"]
            self.data_log["dvl_pitch"] = attitude["pitch"]
            self.data_log["dvl_yaw"] = attitude["yaw"]
            self.data_log["depth"] = attitude["altitude"]

    def getDVLMav(self):
        try:
            dvl_position_delta_data = self.conn.recv_match(type="VISION_POSITION_DELTA", blocking=True)
            if dvl_position_delta_data is not None:
                dvl = dvl_position_delta_data.to_dict()
                print(dvl)
                self.data_log["droll"] = dvl["angle_delta"][0]
                self.data_log["dpitch"] = dvl["angle_delta"][1]
                self.data_log["dyaw"] = dvl["angle_delta"][2]
                self.data_log["dx"] = dvl["position_delta"][0]
                self.data_log["dy"] = dvl["position_delta"][1]
                self.data_log["dz"] = dvl["position_delta"][2]
        except Exception as e:
            print("Cannot get dvl position delta from the mavlink data")
        try:
            dvl_position_data = self.conn.recv_match(type="GLOBAL_VISION_POSITION_ESTIMATE", blocking=True)
            if dvl_position_data is not None:
                dvl = dvl_position_data.to_dict()
                print(dvl)
                self.data_log["roll"] = dvl["roll"]
                self.data_log["pitch"] = dvl["pitch"]
                self.data_log["yaw"] = dvl["yaw"]
                self.data_log["x"] = dvl["x"]
                self.data_log["y"] = dvl["y"]
                self.data_log["z"] = dvl["z"]
        except Exception as e:
            print("Error: ", e)
            print("Cannot get dvl position from the mavlink data")
        try:
            dvl_speed_data = self.conn.recv_match(type="VISION_SPEED_ESTIMATE", blocking=True)
            if dvl_speed_data is not None:
                dvl = dvl_speed_data.to_dict()
                print(dvl)
                self.data_log["vx"] = dvl["x"]
                self.data_log["vy"] = dvl["y"]
                self.data_log["vz"] = dvl["z"]
        except:
            print("Cannot get dvl speed data from the mavlink data")
        try:
            global_origin = self.conn.recv_match(type="SET_GPS_GLOBAL_ORIGIN", blocking=False)
            if global_origin is not None:
                dvl = global_origin.to_dict()
                print(dvl)
                self.data_log["lat"] = dvl["latitude"]
                self.data_log["lon"] = dvl["longitude"]
        except Exception as e:
            print("Error: ", e)
            print("Cannot get global position from DVL")
        # belum ada altitude 
        # https://github.com/bluerobotics/BlueOS-Water-Linked-DVL/blob/master/dvl-a50/mavlink2resthelper.py
        try:
            altitude = self.conn.recv_match(type="DISTANCE_SENSOR", blocking=True)
            if altitude is not None:
                dvl = altitude.to_dict()
                self.data_log["alt"] = dvl["current_distance"]
        except:
            print("Cannot get altitude")
    def getDVLSocket(self):
        #   try receiving data from socket and decode
        try:
            data = self.socket.recv(2048).decode('utf-8')
            if data:
                jsonData= json.loads(data)
                # print(jsonData)
                try:
                    data_list={
                                "altitude":0,
                                "roll":0,
                                "pitch":0,
                                "yaw":0,
                                "x":0,
                                "y":0,
                                "z":0,
                                "vx":0,
                                "vy":0,
                                "vz":0,
                                "velocity_valid":0,
                                "covariance":0,
                                }
                    for key in data_list.keys():
                        if key in jsonData and jsonData[key] is not None:
                            self.data_log[key] = jsonData[key]
                except: 
                    pass
        # data might not be available
        except BlockingIOError:
            pass
        except KeyboardInterrupt:
            self.cleanup()
            raise
        except Exception as e:
            pass

    def cleanup(self):
        if self.socket:
            self.socket.close()
        # if self.target_file:
        #     self.target_file.close()
        print("Socket closed")
        raise
    def getVFRHUD(self):
        vfrData = self.conn.recv_match(type="VFR_HUD", blocking=False)
        if vfrData is not None:
            compass = vfrData.to_dict()

            self.data_log["airspeed(m/s)"] = compass["airspeed"]
            self.data_log["groundspeed(m/s)"] = compass["groundspeed"]
            self.data_log["heading(degree)"] = compass["heading"]
            self.data_log["altitude(m)"] = compass["alt"]
            self.data_log["climbspeed(m/s)"] = compass["climb"]

    def getAccell(self):
        accellData = self.conn.recv_match(type="SCALED_IMU2", blocking=False)
        if accellData is not None:
            accell = accellData.to_dict()
            self.data_log["xacc"] = accell["xacc"] / 100
            self.data_log["yacc"] = accell["yacc"] / 100
            self.data_log["zacc"] = accell["zacc"] / 100
    def get_msg(self, msg_type=None):
        if msg_type is not None:
            data = self.conn.recv_match(type=msg_type, blocking=False)
        else:
            data = self.conn.recv_match()
        if data is not None:
            data = data.to_dict()
            print(data)
    def getDate(self):
        return datetime.datetime.now().date()
    def getTime(self):
        return datetime.datetime.now().time()
    def updateFile(self):
        with open(self.file_name, "r") as target_file:
            data = json.load(target_file)
        data["Date"] = str(self.data_log["Date"])
        data["Time"] = str(self.data_log["Time"])
        data["DVL"]["x"] = self.data_log["y"]
        data["DVL"]["y"] = self.data_log["x"]
        data["DVL"]["z"] = self.data_log["z"]
        data["DVL"]["vx"] = self.data_log["vx"]
        data["DVL"]["vy"] = self.data_log["vy"]
        data["DVL"]["vz"] = self.data_log["vz"]
        data["DVL"]["roll"] = self.data_log["roll"]
        data["DVL"]["pitch"] = self.data_log["pitch"]
        data["DVL"]["yaw"] = self.data_log["yaw"]
        data["DVL"]["alt"] = self.data_log["alt"]
        data["depth"] = self.data_log["depth"]
        with open(self.file_name, "w") as target_file:
            json.dump(data, target_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-timeStep', type=float, default=1, help='addTimestep')
    args = parser.parse_args()
    print("Timestep: ", args.timeStep)
    # update the data
    
    lastlog = time.time()
    logging = Sensors(time_step=args.timeStep)

    while True:
        try:
            logging.getData()
            # time.sleep(0.003)
            # logging.get_msg()
            now = time.time()
            if now - lastlog >= logging.time_step:
                logging.updateFile()
                lastlog=now
                pass
        except KeyboardInterrupt:
            print("Byebye")
            exit()

