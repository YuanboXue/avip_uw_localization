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


# Synchronize orientations
time_values_dvl = dvl_df['timestamp'].values

# Initialize variables
positions_x = [10]
positions_y = [20]
positions_z = [-95]
prev_time = dvl_df.iloc[0]['timestamp']

# Dead reckoning estimation
for index, row in dvl_df.iterrows():
    time = row['timestamp']
    # time = row['field.header.stamp']
    dt = time - prev_time if index > 0 else 0
    dt = dt / 1e9  # Convert nanoseconds to seconds
    vel_x = row['vx']
    vel_y = row['vy']
    vel_z = row['vz']

    velocity_x = positions_x[-1] + vel_x * dt
    velocity_y = positions_y[-1] + vel_y * dt
    velocity_z = positions_z[-1] - vel_z * dt

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

# Save results to TXT in TUM format (time tx ty tz qx qy qz qw)
# with open('dvl_position_vel_0207trail04.txt', 'w') as f:
#     for i in range(len(time_values)):
#         timestamp = time_values[i] / 1e9  # Convert from nanoseconds to seconds
#         tx = positions_x[i+1]
#         ty = positions_y[i+1]
#         tz = positions_z[i+1]
#         qx = 0
#         qy = 0
#         qz = 0
#         qw = 1
#         f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")