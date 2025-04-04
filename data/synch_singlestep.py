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


# Linear interpolation for scalar IMU data
imu_resampled = imu_data.set_index('timestamp').reindex(dvl_data['timestamp']).interpolate().reset_index()

# Slerp interpolation for quaternion data
imu_resampled[['field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w']] = slerp(
    imu_data['timestamp'].values.astype(np.float64),
    imu_data[['field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w']].values,
    dvl_data['timestamp'].values.astype(np.float64)
)

# Find the nearest pressure sensor data for each DVL timestamp
pressure_nearest = pressure_data.set_index('timestamp').reindex(dvl_data['timestamp'], method='nearest').reset_index()

# Find the nearest ground truth data for each DVL timestamp
ground_truth_nearest = ground_truth_data.set_index('timestamp').reindex(dvl_data['timestamp'], method='nearest').reset_index()

# Combine all synchronized data into a single DataFrame
# synchronized_data = pd.concat([dvl_data, imu_resampled.drop(columns=['timestamp']), pressure_nearest.drop(columns=['timestamp']), ground_truth_nearest.drop(columns=['timestamp'])], axis=1)

# Select critical information
synchronized_data = pd.DataFrame({
    'timestamp': dvl_data['%time'],
    # DVL linear velocities
    'dvl_velocity_x': dvl_data['field.velocity.x'],
    'dvl_velocity_y': dvl_data['field.velocity.y'],
    'dvl_velocity_z': dvl_data['field.velocity.z'],
    # IMU orientations
    'imu_orientation_x': imu_resampled['field.orientation.x'],
    'imu_orientation_y': imu_resampled['field.orientation.y'],
    'imu_orientation_z': imu_resampled['field.orientation.z'],
    'imu_orientation_w': imu_resampled['field.orientation.w'],
    # IMU angular velocities
    'imu_angular_velocity_x': imu_resampled['field.angular_velocity.x'],
    'imu_angular_velocity_y': imu_resampled['field.angular_velocity.y'],
    'imu_angular_velocity_z': imu_resampled['field.angular_velocity.z'],
    # IMU linear accelerations
    'imu_linear_acceleration_x': imu_resampled['field.linear_acceleration.x'],
    'imu_linear_acceleration_y': imu_resampled['field.linear_acceleration.y'],
    'imu_linear_acceleration_z': imu_resampled['field.linear_acceleration.z'],
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

# Save the synchronized data to a new CSV file
synchronized_data.to_csv('synchronized_data.csv', index=False)

print("Synchronized data has been saved to 'synchronized_data.csv'.")
