import numpy as np
import math
# from scipy.interpolate import CubicSpline
# import time

class Guidance:
    def __init__(self, waypoints, mode, lookahead_distance=None, k_e=None, distance_treshold = 0.1, generate_virtual_wp=False, dt=1):
        self.waypoints = waypoints
        self.mode = mode  # 'LOS', 'PP', or 'Stanley'
        self.lookahead_distance = lookahead_distance  # Only for PP
        self.k_e = k_e  # Only for Stanley
        self.current_waypoint_index = 0
        self.distance_treshold = distance_treshold
        self.dt = dt
        self.max_steering_angle = np.radians(40)
        self.update_status()
        self.distance_to_wp = 0
        if generate_virtual_wp:
            self.generate_virtual_waypoints()

    def update_status(self):
        self.status="Going to WP: " + str(self.current_waypoint_index)

    # Update waypoint index when close enough to the current waypoint
    def update_waypoint(self, vehicle_position):
        current_wp = self.waypoints[self.current_waypoint_index]
        # calculate the distance
        distance_to_wp = np.linalg.norm(np.array(vehicle_position) - np.array(current_wp))
        # check the distance
        print(f"distance to WP {self.current_waypoint_index}: ",distance_to_wp)
        self.update_status()
        if distance_to_wp < self.distance_treshold:
            # if its not the last waypoint, go to next waypoint by increase the index
            if self.current_waypoint_index < (len(self.waypoints) -1):
                self.current_waypoint_index += 1
                print("next_wp")
            # else, done
            else:
                self.status="Guidance Done"
        self.distance_to_wp = distance_to_wp
        return distance_to_wp

    # Line-of-Sight Guidance
    def los(self, vehicle_position, vehicle_heading):
        # get the way point
        self.update_waypoint(vehicle_position)
        target_point = self.waypoints[self.current_waypoint_index]
        dx = target_point[0] - vehicle_position[0]
        dy = target_point[1] - vehicle_position[1]
        
        target_heading = (np.arctan2(dx, dy))
        # print("Here is the guidance")
        # print(dx,dy)
        # print(target_heading)
        heading_error = target_heading-vehicle_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        steering_angle=np.clip(heading_error,-self.max_steering_angle,self.max_steering_angle)
        target_heading = vehicle_heading+steering_angle
        return target_heading, target_point, self.status

    def generate_virtual_waypoints(self, num_virtual_points=10):
        virtual_waypoints = []
        waypoints=self.waypoints
        print(waypoints)
        for i in range(len(waypoints) - 1):
            # Add the current main waypoint
            virtual_waypoints.append(waypoints[i])
            # Linearly interpolate points between current and next waypoint
            for j in range(1, num_virtual_points + 1):
                wpx=waypoints[i][0]
                next_wpx = waypoints[i+1][0]
                wpy=waypoints[i][1]
                next_wpy = waypoints[i+1][1]
                # add the point between the 2 points
                interp_point_x = wpx + (next_wpx- wpx) * (j/(num_virtual_points + 1))
                interp_point_y = wpy + (next_wpy- wpy) * (j/(num_virtual_points+1))
                virtual_waypoints.append((interp_point_x,interp_point_y))
        # Add the last main waypoint
        virtual_waypoints.append(waypoints[-1])
        self.waypoints = virtual_waypoints
    
    # Pure Pursuit Guidance
    def pure_pursuit(self, vehicle_position, vehicle_heading, velocity):
        # Find the target point by checking the lookahead distance
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            distance = np.linalg.norm(np.array(vehicle_position) - np.array(self.waypoints[i]))
            # print("Distance: ", distance, "WP: ", self.waypoints[i])
            # print("Checking")
            if distance >= self.lookahead_distance:
                # print("Ketemu bos!", i)
                self.current_waypoint_index=i
                break

        target_point = self.waypoints[self.current_waypoint_index]
        # print("Mantap Dapat target point")
        # x and y position error
        dx = target_point[0] - vehicle_position[0]
        dy = target_point[1] - vehicle_position[1]
        # distance to wp
        distance = (dx**2 + dy**2)**0.5
        # heading required to waypoint

        # Calculate the heading required to reach the waypoint (with 0° as north, positive counterclockwise)
        target_heading = np.arctan2(dx, dy)

        # Calculate the heading error
        heading_error = target_heading - vehicle_heading
        # Normalize the heading error to the range [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        # Calculate the heading error between the vehicle's heading and the target heading
        
        # reference
        # https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9193967
        # becareful when modifying wheel_base, could result in bad steering angle
        wheel_base= 0.1
        steering_angle = np.arctan2(2 * wheel_base * np.sin(heading_error), self.lookahead_distance)
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        desired_heading = vehicle_heading + steering_angle

        if self.current_waypoint_index == (len(self.waypoints)-1):
            self.status="Guidance Done"

        return desired_heading, target_point, self.status

    # Stanley Controller Guidance
    def stanley(self, vehicle_position, vehicle_heading, velocity):
        self.update_waypoint(vehicle_position)
        target_point = self.waypoints[self.current_waypoint_index]
        prev_target_point = self.waypoints[self.current_waypoint_index-1] if self.current_waypoint_index != 0 else vehicle_position
        # calculate the distance error to path
        cross_track_error, _ = self.calculate_cross_track(prev_target_point, target_point, vehicle_position)
        
        # calculate the path heading
        dx = target_point[0] - vehicle_position[0]
        dy = target_point[1] - vehicle_position[1]
        distance = (dx**2 + dy**2)**0.5
        path_heading = np.arctan2(dx, dy)
        # heading error = path_heading-vehicle
        heading_error = path_heading-vehicle_heading
        # correct the heading error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # the cross track correction = tan-1(k*E_crosstrack/velocity)
        # https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
        cross_track_correction = np.arctan2(self.k_e * cross_track_error, velocity)

        steering_angle = heading_error - cross_track_correction
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        desired_heading = vehicle_heading + steering_angle
        forward_velo = self.calc_velo(0.5, distance, 2)
        if heading_error > 0.4:
            forward_velo = 0
        
        lateral_velo = self.calc_velo(0.5, cross_track_error, 0.5)
        # debugging code
        # print("vehicle_heading", np.degrees(vehicle_heading))
        print("heading_error", np.degrees(heading_error))
        # print("cross_track_correction", np.degrees(cross_track_correction))
        # print("Prev target point", prev_target_point)
        print("Target point", target_point)
        # print("Vehicle Position",vehicle_position)
        # print("Cross_track_error",cross_track_error)
        # print("Closest Point on Path",_)
         # print("path_heading", np.degrees(path_heading))

        return desired_heading, target_point, self.status, forward_velo, lateral_velo
    
    def calculate_cross_track(self, waypoint1, waypoint2, current_position):
        """
        Calculate the minimum cross-track error between the current position and the path defined by two waypoints.
        
        Parameters:
        - waypoint1: Starting waypoint of the path (x, y).
        - waypoint2: Ending waypoint of the path (x, y).
        - current_position: The current vehicle position (x, y).
        
        Returns:
        - cross_track_error: Minimum perpendicular distance from the current position to the path.
        - closest_point_on_path: The point on the path that is closest to the current position.
        """
        
        # Convert inputs to numpy arrays
        wp1 = np.array(waypoint1)
        wp2 = np.array(waypoint2)
        pos = np.array(current_position)
        
        # Vector path from waypoint1 to waypoint2
        path_vector = wp2 - wp1
        
        # Vector from current position to waypoint1
        position_vector = pos - wp1
        
        # Normalize the path vector
        path_vector_length = np.linalg.norm(path_vector)
        # checking the vector_length
        if path_vector_length!=0:
            path_vector_normalized = path_vector / path_vector_length
            # cross length = |a|.sin alpha = a x b /|b|)
            cross_length2 = np.cross(position_vector,path_vector) / path_vector_length
        else:
            path_vector_normalized =0
            cross_length2=0
        
        # Dot product to find the projection length (along the path)
        projection_length = np.dot(position_vector, path_vector_normalized)
        
        # # Calculate the closest point on the path
        closest_point_on_path = wp1 + projection_length * path_vector_normalized
        
        # # Cross-track error: distance from current position to closest point on path
        # cross_track_error = np.linalg.norm(pos - closest_point_on_path)
        
        return cross_length2, closest_point_on_path
        
    def calc_velo(self, k, error_now, max_velo):
        # sigmoid function
        # https://andymath.com/logistic-function/
        sigmoid = 2/(1+math.exp(-k*error_now))-1
        return sigmoid*max_velo
        
    def calculate_steering(self, vehicle_position, vehicle_heading, velocity=0.5):
        if self.mode == 'LOS':
            return self.los(vehicle_position, vehicle_heading)
        elif self.mode == 'PP':
            return self.pure_pursuit(vehicle_position, vehicle_heading, velocity)
        elif self.mode == 'Stanley':
            return self.stanley(vehicle_position, vehicle_heading, velocity)
        
