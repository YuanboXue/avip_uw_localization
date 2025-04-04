import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dvl data from csv
dvl_df = pd.read_csv('dvl_davepool.csv')

gt_df = pd.read_csv('gt_davepool.csv')
gt_position_x = gt_df['x'].values
gt_position_y = gt_df['y'].values
gt_position_z = gt_df['z'].values
gt_time = gt_df['timestamp'].values
gt_quaternion_x = gt_df['field.pose.pose.orientation.x'].values
gt_quaternion_y = gt_df['field.pose.pose.orientation.y'].values
gt_quaternion_z = gt_df['field.pose.pose.orientation.z'].values
gt_quaternion_w = gt_df['field.pose.pose.orientation.w'].values

imu_df = pd.read_csv('imu_davepool.csv')
imu_time = imu_df['timestamp'].values

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
imu_quaternions = imu_df[['field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w']].values

# Synchronize orientations
# time_values_dvl = dvl_df['timestamp'].values

# Initialize variables
positions_x = [10]
positions_y = [20]
positions_z = [-95]
prev_time = dvl_df.iloc[0]['timestamp']

# Dead reckoning estimation
for index, row in dvl_df.iterrows():
    time = row['timestamp']
    dt = time - prev_time if index > 0 else 0
    dt = dt / 1e9  # Convert nanoseconds to seconds
    vel_x = row['vx']
    vel_y = row['vy']
    vel_z = row['vz']

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
time_values = np.array(dvl_df['timestamp'])  # Convert time values to numpy array
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