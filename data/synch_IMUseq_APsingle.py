import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Load the data from CSV files
imu_data = pd.read_csv('imu_davepool_shr_train01.csv')
dvl_data = pd.read_csv('dvl_davepool_shr_train01.csv')
pressure_data = pd.read_csv('ps_davepool_shr_train01.csv')
ground_truth_data = pd.read_csv('gt_davepool_shr_train01.csv')

# Convert timestamps to datetime
imu_data['timestamp'] = pd.to_datetime(imu_data['%time'])
dvl_data['timestamp'] = pd.to_datetime(dvl_data['%time'])
pressure_data['timestamp'] = pd.to_datetime(pressure_data['%time'])
ground_truth_data['timestamp'] = pd.to_datetime(ground_truth_data['%time'])

# Function to perform slerp interpolation on quaternions
def slerp(timestamps, quaternions, target_timestamps):
    key_rots = R.from_quat(quaternions)
    slerp_interpolator = Slerp(timestamps, key_rots)
    interp_rots = slerp_interpolator(target_timestamps)
    return interp_rots.as_quat()

# Extract a sequence of IMU data around each DVL timestamp
sequence_length = 10  # Number of IMU time steps to extract
imu_sequences = []

for dvl_time in dvl_data['timestamp']:
    # Find the closest IMU data points around the DVL timestamp
    imu_window = imu_data[(imu_data['timestamp'] >= dvl_time - pd.Timedelta(seconds=0.5)) & 
                          (imu_data['timestamp'] <= dvl_time + pd.Timedelta(seconds=0.5))]
    
    # If there are not enough data points, interpolate to fill the sequence
    if len(imu_window) < sequence_length:
        imu_window = imu_window.set_index('timestamp').reindex(
            pd.date_range(start=dvl_time - pd.Timedelta(seconds=0.5), 
                          end=dvl_time + pd.Timedelta(seconds=0.5), 
                          periods=sequence_length)
        ).interpolate().reset_index()
    
    # Ensure the sequence is exactly 10 time steps
    imu_window = imu_window.head(sequence_length)
    
    imu_sequences.append(imu_window)

# Save the resampled IMU data sequences to a CSV file
imu_sequences_df = pd.concat(imu_sequences, keys=dvl_data['timestamp'], names=['dvl_timestamp', 'sequence_index'])
imu_sequences_df.to_csv('imu_sequences.csv', index=False)

# Find the nearest pressure sensor data for each DVL timestamp
pressure_nearest = pressure_data.set_index('timestamp').reindex(dvl_data['timestamp'], method='nearest').reset_index()

# Find the nearest ground truth data for each DVL timestamp
ground_truth_nearest = ground_truth_data.set_index('timestamp').reindex(dvl_data['timestamp'], method='nearest').reset_index()

# Combine all synchronized data into a single DataFrame for non-sequential data
synchronized_data = pd.DataFrame({
    'timestamp': dvl_data['%time'],
    # DVL linear velocities
    'dvl_velocity_x': dvl_data['field.velocity.x'],
    'dvl_velocity_y': dvl_data['field.velocity.y'],
    'dvl_velocity_z': dvl_data['field.velocity.z'],
    # Pressure sensor fluid pressure
    'pressure_fluid_pressure': pressure_nearest['field.fluid_pressure'],
    # Ground truth positions
    'gt_position_x': ground_truth_nearest['field.pose.pose.position.x'],
    'gt_position_y': ground_truth_nearest['field.pose.pose.position.y'],
    'gt_position_z': ground_truth_nearest['field.pose.pose.position.z'],
    # Ground truth orientations
    'gt_orientation_x': ground_truth_nearest['field.pose.pose.orientation.x'],
    'gt_orientation_y': ground_truth_nearest['field.pose.pose.orientation.y'],
    'gt_orientation_z': ground_truth_nearest['field.pose.pose.orientation.z'],
    'gt_orientation_w': ground_truth_nearest['field.pose.pose.orientation.w'],
    # Ground truth linear velocities
    'gt_linear_acceleration_x': ground_truth_nearest['field.twist.twist.linear.x'],
    'gt_linear_acceleration_y': ground_truth_nearest['field.twist.twist.linear.y'],
    'gt_linear_acceleration_z': ground_truth_nearest['field.twist.twist.linear.z'],
    # Ground truth angular velocities
    'gt_angular_velocity_x': ground_truth_nearest['field.twist.twist.angular.x'],
    'gt_angular_velocity_y': ground_truth_nearest['field.twist.twist.angular.y'],
    'gt_angular_velocity_z': ground_truth_nearest['field.twist.twist.angular.z']
})

# Save the synchronized non-sequential data to a new CSV file
synchronized_data.to_csv('synchronized_data.csv', index=False)

print("IMU sequences have been saved to 'imu_sequences.csv'.")
print("Synchronized data has been saved to 'synchronized_data.csv'.")