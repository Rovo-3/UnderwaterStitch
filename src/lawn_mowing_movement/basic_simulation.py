import math
from guidance import Guidance
from waypoint_generator import SweepWPGenerator
import numpy as np
import matplotlib.pyplot as plt

# Simulation
def simulate_guidance(waypoints, mode, lookahead_distance=None, k_e=None, initial_position=(0, 0), initial_heading=0.0):
    vehicle_position = initial_position
    # heading in pi
    initial_heading*=math.pi/180
    vehicle_heading = initial_heading
    dt = 0.1
    velocity = 0.1

    guidance = Guidance(waypoints, mode, lookahead_distance, k_e, dt)

    trajectory = [vehicle_position]
    heading = [(0,vehicle_heading)]

    for t in np.arange(1, 1000, dt):
        # print(vehicle_heading)
        # print(guidance.current_waypoint_index)
        target_angle, target_point = guidance.calculate_steering(vehicle_position, vehicle_heading, velocity)

        print(target_point)

        # Update vehicle heading
        vehicle_heading = target_angle
        vehicle_heading = (vehicle_heading + math.pi) % (2*math.pi) - math.pi
        # turn to pi
        # vehicle_heading *= math.pi/180
        
        # Update vehicle position
        vehicle_position = (
            vehicle_position[0] + velocity * np.sin(vehicle_heading) * dt,
            vehicle_position[1] + velocity * np.cos(vehicle_heading) * dt
        )

        # Store trajectory
        trajectory.append(vehicle_position)
        heading.append((t,vehicle_heading))

        # Stop if near the last waypoint
        if np.linalg.norm(np.array(vehicle_position) - np.array(waypoints[-1])) < 0.2:
            break
    
    return trajectory, heading, guidance

# Generate waypoints
wp = SweepWPGenerator(length=3, angle=90, gap=1, iteration=2)
waypoints = wp.generate()
# waypoints = [(0,0),(3,3)]

# Simulate for each guidance method
lookahead_distance = 0.2  # Pure Pursuit Lookahead Distance
k_e = 0.4  # Stanley Gain

initial_position = (-1, -1)
initial_heading = 30.0

# Simulate LOS
los_trajectory, los_heading, _ = simulate_guidance(waypoints, 'LOS', initial_position=initial_position, initial_heading=initial_heading)
print(los_heading)
# Simulate Pure Pursuit
pp_trajectory, pp_heading, guidance = simulate_guidance(waypoints, 'PP', lookahead_distance=lookahead_distance, initial_position=initial_position, initial_heading=initial_heading)

# Simulate Stanley
stanley_trajectory, stanley_heading, _ = simulate_guidance(waypoints, 'Stanley', k_e=k_e, initial_position=initial_position, initial_heading=initial_heading)

# Plot results
waypoints = np.array(waypoints)
los_trajectory = np.array(los_trajectory)
los_heading=np.array(pp_heading)
pp_trajectory = np.array(pp_trajectory)
# virtual_wp = np.array(guidance.virtual_waypoints)
stanley_trajectory = np.array(stanley_trajectory)

plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', label='Waypoints')
# plt.plot(virtual_wp[:, 0], virtual_wp[:, 1], 'ro-', label='Waypoints')
plt.plot(los_trajectory[:, 0], los_trajectory[:, 1], 'g-', label='LOS Trajectory')
plt.plot(pp_trajectory[:, 0], pp_trajectory[:, 1], 'b-', label='Pure Pursuit Trajectory')
plt.plot(stanley_trajectory[:, 0], stanley_trajectory[:, 1], 'm-', label='Stanley Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('Guidance Result')


plt.figure()
plt.plot(los_heading[:, 0], los_heading[:, 1], 'g', label='LOS Heading')
plt.show()
