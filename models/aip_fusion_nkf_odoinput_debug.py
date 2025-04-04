import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Extract relevant columns
timestamps = data['timestamp'].values
dvl_velocities = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
imu_orientations = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w']].values
pressure_data = data['pressure_fluid_pressure'].values
gt_positions = data[['gt_position_x', 'gt_position_y', 'gt_position_z']].values

# gt_positions[:, 2] = - gt_positions[:, 2] - 95
# gt_positions[:, 2] = gt_positions[:, 2] + 95

# Initialize variables for estimated positions
positions_x = [10]
positions_y = [20]
# positions_z = [0.2111]
# positions_z = [0.1051]
positions_z = [-94.78894713416668]
prev_time = timestamps[0]

# Helper function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    rotation = R.from_quat(q)
    return rotation.as_matrix()

# Constants
gravity = 9.80665  # m/s^2
water_density = 1000  # kg/m^3
# Calculate water depth from pressure sensor
# water_depth = pressure_data / (water_density * gravity)
water_depth = - (pressure_data * 1000 - 101325)/(1000*9.80665)
print("water_depth", water_depth)

# Dead reckoning to estimate position and orientation
# estimated_position = np.zeros((len(dvl_velocity), 3))
# estimated_orientation = imu_orientation.copy()

# Start point coordinates
# estimated_position[0] = [10, 20, -95]

# Dead reckoning with IMU and DVL fusion
for i in range(1, len(timestamps)):
    # Time difference (dt)
    current_time = timestamps[i]
    dt = (current_time - prev_time) / 1e9  # Convert nanoseconds to seconds

    # Get DVL velocities
    dvl_velocity = dvl_velocities[i]

    # Get IMU orientation and convert to rotation matrix
    imu_orientation = imu_orientations[i]
    rotation_matrix = quaternion_to_rotation_matrix(imu_orientation)

    # Rotate DVL velocities into the global frame
    global_velocity = rotation_matrix @ dvl_velocity

    # Update positions using velocity integration
    velocity_x = positions_x[-1] + global_velocity[0] * dt
    velocity_y = positions_y[-1] + global_velocity[1] * dt
    # velocity_z = positions_z[-1] + global_velocity[2] * dt
    velocity_z = water_depth[i]  # Replace z-position with water depth

    positions_x.append(velocity_x)
    positions_y.append(velocity_y)
    positions_z.append(velocity_z)

    # Update previous time
    prev_time = current_time

# Convert positions to numpy arrays
positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
positions_z = np.array(positions_z)

# positions_z = positions_z - positions_z[0]
# gt_positions[:, 2] = gt_positions[:, 2] - gt_positions[0, 2]

# Plot the estimated trajectory and ground truth
plt.figure(figsize=(10, 6))
plt.plot(positions_x, positions_y, label='Estimated Trajectory (IMU + DVL)', color='blue')
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.scatter(gt_positions[0, 0], gt_positions[0, 1], color='green', label='Start (Ground Truth)', zorder=5)
plt.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color='red', label='End (Ground Truth)', zorder=5)
plt.scatter(positions_x[0], positions_y[0], color='purple', label='Start (Estimated)', zorder=5)
plt.scatter(positions_x[-1], positions_y[-1], color='brown', label='End (Estimated)', zorder=5)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectory Estimated by Fusing IMU and DVL')
plt.legend()
plt.grid()
plt.show()

# Plot the estimated trajectory in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_x, positions_y, positions_z, label='Estimated Trajectory (IMU + DVL)', color='blue')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], color='green', label='Start (Ground Truth)', zorder=5)
ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], color='red', label='End (Ground Truth)', zorder=5)
ax.scatter(positions_x[0], positions_y[0], positions_z[0], color='purple', label='Start (Estimated)', zorder=5)
ax.scatter(positions_x[-1], positions_y[-1], positions_z[-1], color='brown', label='End (Estimated)', zorder=5)
ax.set_xlabel('X Position (m)') 
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Vehicle Trajectory Estimated by Fusing IMU and DVL')
ax.legend()
plt.show()

# Plot the comparison of estimated water-depth and ground truth positions in z axis
# plt.figure(figsize=(10, 6))
# plt.plot(water_depth, label='Estimated Water Depth', color='blue')
# plt.plot(gt_positions[:, 2], label='Ground Truth Z Position', linestyle='--', color='orange')
# # plt.scatter(gt_positions[0, 2], color='green', label='Start (Ground Truth)', zorder=5)
# # plt.scatter(gt_positions[-1, 2], color='red', label='End (Ground Truth)', zorder=5)
# # plt.scatter(positions_z[0], color='purple', label='Start (Estimated)', zorder=5)    
# # plt.scatter(positions_z[-1], color='brown', label='End (Estimated)', zorder=5)
# plt.xlabel('Time Step')
# plt.ylabel('Z Position (m)')
# plt.title('Estimated Water Depth and Ground Truth Z Position')
# plt.legend()
# plt.grid()
# plt.show()
