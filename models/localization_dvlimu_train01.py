import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Extract the relevant columns
imu_time = data['timestamp'].values
imu_orientation = data[['imu_orientation_w', 'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z']].values
imu_linear_acc = data[['imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z']].values
imu_angular_vel = data[['imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z']].values
dvl_data = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
pressure_data = data['pressure_fluid_pressure'].values
ground_truth = data[['gt_position_x', 'gt_position_y', 'gt_position_z', 'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values
gt_position_x = data['gt_position_x'].values
gt_position_y = data['gt_position_y'].values
gt_position_z = data['gt_position_z'].values
gt_time = data['timestamp'].values
gt_quaternion_w = data['gt_orientation_w'].values
gt_quaternion_x = data['gt_orientation_x'].values
gt_quaternion_y = data['gt_orientation_y'].values
gt_quaternion_z = data['gt_orientation_z'].values

# Helper function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

# Synchronize IMU orientation with DVL timestamps
def synchronize_imu(dvl_time, imu_time, imu_quaternions):
    idx = np.searchsorted(imu_time, dvl_time, side='left')
    if idx > 0 and (idx == len(imu_time) or abs(dvl_time - imu_time[idx-1]) < abs(dvl_time - imu_time[idx])):
        idx -= 1
    return imu_quaternions[idx]

# Extract IMU quaternion data
imu_quaternions = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w']].values

# Synchronize orientations
# time_values_dvl = dvl_df['timestamp'].values

# Initialize variables
positions_x = [10]
positions_y = [20]
positions_z = [-95]
prev_time = data.iloc[0]['timestamp']

# Dead reckoning estimation
for index, row in data.iterrows():
    time = row['timestamp']
    dt = time - prev_time if index > 0 else 0
    dt = dt / 1e9  # Convert nanoseconds to seconds
    vel_x = row['dvl_velocity_x']
    vel_y = row['dvl_velocity_y']
    vel_z = row['dvl_velocity_z']

    # velocity_x = positions_x[-1] + vel_x * dt
    # velocity_y = positions_y[-1] + vel_y * dt
    # velocity_z = positions_z[-1] - vel_z * dt

    # Synchronize IMU orientation
    imu_quat = synchronize_imu(time, imu_time, imu_quaternions)
    rotation_matrix = quaternion_to_rotation_matrix(imu_quat)

    # Rotate DVL velocities into the global frame
    dvl_velocity = np.array([vel_x, vel_y, vel_z])
    global_velocity = rotation_matrix @ dvl_velocity

    # Update positions
    velocity_x = positions_x[-1] + global_velocity[0] * dt
    velocity_y = positions_y[-1] + global_velocity[1] * dt
    velocity_z = positions_z[-1] + global_velocity[2] * dt

    positions_x.append(velocity_x)
    positions_y.append(velocity_y)
    positions_z.append(velocity_z)
    
    prev_time = time

positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
positions_z = np.array(positions_z)

print("First position:", positions_x[0], positions_y[0])

# Plotting the trajectory
plt.figure(figsize=(10, 6))
time_values = np.array(data['timestamp'])  # Convert time values to numpy array
# plt.plot(positions_x[1:], positions_y[1:], label='DVL')
plt.plot(positions_x, positions_y, label='DVL Vel')
plt.plot(gt_position_x, gt_position_y, label='Ground Truth', linestyle='--')
plt.plot(gt_position_x[0], gt_position_y[0], 'go', label='Start GT')    
plt.plot(gt_position_x[-1], gt_position_y[-1], 'ro', label='End GT')
# Start and end points
plt.plot(positions_x[0], positions_y[0], 'go', label='Start')
plt.plot(positions_x[-1], positions_y[-1], 'ro', label='End')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory Estimated by Dead Reckoning')
plt.legend()

plt.show()