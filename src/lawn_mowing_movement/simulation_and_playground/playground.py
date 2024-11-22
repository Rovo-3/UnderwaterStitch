# l = [0,1,2,3,4,5,6,7,8,9]
# print(l[0:5])
# import numpy as np
# import time
# vehicle_position=[(0,0)]
# current_wp = [(10,10)]

# fstcalc = time.time()
# distance_to_wp = np.linalg.norm(np.array(vehicle_position) - np.array(current_wp))
# time_elapsed1=time.time()-fstcalc

# scndcalc = time.time()
# distance_calc = ((vehicle_position[0][0]-current_wp[0][0])**2 + (vehicle_position[0][1]-current_wp[0][1])**2)**(0.5)
# time_elapsed2 = time.time()-scndcalc
# print(distance_to_wp,time_elapsed1) #faster for big data
# print(distance_calc, time_elapsed2) #faster for small data



        # if not hasattr(self, 'virtual_waypoints'):
        #     self.virtual_waypoints = self.generate_virtual_waypoints()
        #     self.waypoints = self.virtual_waypoints

        # # Find the target point by checking the lookahead distance
        # for i in range(0, len(self.waypoints)):
        #     distance = np.linalg.norm(np.array(vehicle_position) - np.array(self.waypoints[i]))
        #     print("Distance: ", distance, "WP: ", self.waypoints[i])
        #     if distance >= self.lookahead_distance:
        #         target_point = self.virtual_waypoints[i]
        #         break


import numpy as np

# def calculate_cross_track(waypoint1, waypoint2, current_position):
#     """
#     Calculate the minimum cross-track error between the current position and the path defined by two waypoints.
    
#     Parameters:
#     - waypoint1: Starting waypoint of the path (x, y).
#     - waypoint2: Ending waypoint of the path (x, y).
#     - current_position: The current vehicle position (x, y).
    
#     Returns:
#     - cross_track_error: Minimum perpendicular distance from the current position to the path.
#     - closest_point_on_path: The point on the path that is closest to the current position.
#     """
    
#     # Convert inputs to numpy arrays
#     wp1 = np.array(waypoint1)
#     wp2 = np.array(waypoint2)
#     pos = np.array(current_position)
    
#     # Vector path from waypoint1 to waypoint2
#     path_vector = wp2 - wp1
    
#     # Vector from current position to waypoint1
#     position_vector = pos - wp1
    
#     # Normalize the path vector
#     path_vector_normalized = path_vector / np.linalg.norm(path_vector)

#     # cross length = sin alpha * position vector length.
#     # alpha = arc cos (a.b/|a|.|b|)
#     # cross length = |a|.sin alpha = a x b /|b|)
#     alpha = np.arccos(np.dot(position_vector, path_vector)/(np.linalg.norm(position_vector)*np.linalg.norm(path_vector)))
#     position_vector_length = np.linalg.norm(position_vector)

#     cross_length1 = np.sin(alpha)*position_vector_length
#     cross_length2 = np.cross(position_vector,path_vector) / np.linalg.norm(path_vector)

#     # Dot product to find the projection length (along the path)
#     projection_length = np.dot(position_vector, path_vector_normalized)
    
#     # Calculate the closest point on the path
#     closest_point_on_path = wp1 + projection_length * path_vector_normalized
    
#     # Cross-track error: distance from current position to closest point on path
#     cross_track_error = np.linalg.norm(pos - closest_point_on_path)
    
#     return cross_track_error, cross_length1, cross_length2, closest_point_on_path


# # Example usage
# waypoint1 = (0, 0)  # First waypoint (x1, y1)
# waypoint2 = (0, 10)  # Second waypoint (x2, y2)
# current_position = (0.5585122305476454, -0.29544232590402003)  # Current vehicle position (x, y)

# cross_track_error, c1, c2, closest_point_on_path = calculate_cross_track(waypoint1, waypoint2, current_position)

# print(f"Cross-track error: {cross_track_error:.2f}")
# print(f"Cross-track error1: {c1:.2f}")
# print(f"Cross-track error2: {c2:.2f}")
# print(f"Closest point on path: {closest_point_on_path}")
import os
file_path = os.path.abspath(__file__)

print(file_path)

import os

# Get the directory of the current Python file
file_directory = os.path.dirname(os.path.abspath(__file__))

print("Directory:", file_directory)