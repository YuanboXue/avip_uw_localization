import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load the data
imu_data = pd.read_csv('imu_davepool_strai.csv')
dvl_data = pd.read_csv('dvl_davepool_strai.csv')

# Convert timestamps to a common format if necessary
imu_data['timestamp'] = pd.to_datetime(imu_data['%time'])
dvl_data['timestamp'] = pd.to_datetime(dvl_data['timestamp'])

# Interpolate DVL data to match IMU timestamps
dvl_interp = interp1d(dvl_data['timestamp'].astype(np.int64), dvl_data[['vx', 'vy', 'vz']], axis=0, fill_value="extrapolate")
dvl_resampled = dvl_interp(imu_data['timestamp'].astype(np.int64))

# Initialize state vector [x, y, z, vx, vy, vz, qx, qy, qz, qw]
state = np.zeros(10)

# Initialize covariance matrix
P = np.eye(10)

# Define process and measurement noise
Q = np.eye(10) * 0.01  # Process noise
R_measurement = np.eye(6) * 0.1    # Measurement noise

# Kalman Filter functions
def predict(state, P, imu_measurement, dt):
    # Extract quaternion and convert to rotation matrix
    q = [imu_measurement['field.orientation.x'], imu_measurement['field.orientation.y'], imu_measurement['field.orientation.z'], imu_measurement['field.orientation.w']]
    rotation = R.from_quat(q).as_matrix()

    # Update state transition
    F = np.eye(10)
    F[0:3, 3:6] = np.eye(3) * dt

    # Predict next state
    state[0:3] += state[3:6] * dt  # Update position
    state[3:6] = rotation @ state[3:6]  # Update velocity based on orientation

    # Update covariance
    P = F @ P @ F.T + Q
    return state, P

def update(state, P, measurement):
    # Measurement matrix
    H = np.zeros((6, 10))
    H[0:3, 0:3] = np.eye(3)  # Position
    H[3:6, 3:6] = np.eye(3)  # Velocity

    # Measurement residual
    z = np.hstack((measurement['x'], measurement['y'], measurement['z'], measurement['vx'], measurement['vy'], measurement['vz']))
    y = z - H @ state

    # Kalman gain
    S = H @ P @ H.T + R_measurement
    K = P @ H.T @ np.linalg.inv(S)

    # Update state and covariance
    state = state + K @ y
    P = (np.eye(10) - K @ H) @ P
    return state, P

# Main loop
estimated_positions = []

for i, imu_row in imu_data.iterrows():
    dt = 0.01  # Assuming a constant time step, adjust as necessary

    # Predict step
    state, P = predict(state, P, imu_row, dt)

    # Update step with interpolated DVL data
    dvl_measurement = {
        'x': 0, 'y': 0, 'z': 0,  # Assuming no direct position measurement from DVL
        'vx': dvl_resampled[i, 0],
        'vy': dvl_resampled[i, 1],
        'vz': dvl_resampled[i, 2]
    }
    state, P = update(state, P, dvl_measurement)

    # Print or store the state for analysis
    # print(f"Time: {imu_row['timestamp']}, State: {state}")

    # Store the estimated position
    estimated_positions.append(state[0:3])

# Convert to numpy array for plotting
estimated_positions = np.array(estimated_positions)

# Plot the estimated trajectory
plt.figure(figsize=(10, 6))
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Estimated Trajectory')
plt.legend()
plt.show()