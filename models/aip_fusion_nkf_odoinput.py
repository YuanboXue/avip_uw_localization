import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# Load the synchronized CSV file
# 1) for each DVL timestamp, use interpolation to resample the IMU data; 
# 2) for each DVL timestamp, find the nearest pressure sensor data; 
# 3) for each DVL timestamp, find the nearest ground truth as the train label
data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Extract the relevant columns
timestamps = data['timestamp'].values
dvl_velocities = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
imu_orientations = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w']].values
pressure_data = data['pressure_fluid_pressure'].values
gt_positions = data[['gt_position_x', 'gt_position_y', 'gt_position_z']].values
gt_orientations = data[['gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

# Helper function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    rotation = R.from_quat(q)
    return rotation.as_matrix()

# Calculate water depth from pressure sensor
water_depth = - (pressure_data * 1000 - 101325)/(1000*9.80665)

# Initialize variables for estimated positions
positions_x = [10]
positions_y = [20]
positions_z = [-94.78894713416668]
prev_time = timestamps[0]

# Dead reckoning fusing IMU, DVL, and Pressure Sensor
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

# Plot the estimated trajectory and ground truth
plt.figure(figsize=(10, 6))
plt.plot(positions_x, positions_y, label='Estimated Trajectory fusing DVL, IMU and PS', color='blue')
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.scatter(gt_positions[0, 0], gt_positions[0, 1], color='green', label='Start (Ground Truth)', zorder=5)
plt.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color='red', label='End (Ground Truth)', zorder=5)
plt.scatter(positions_x[0], positions_y[0], color='purple', label='Start (Estimated)', zorder=5)
plt.scatter(positions_x[-1], positions_y[-1], color='brown', label='End (Estimated)', zorder=5)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectory Estimated by Fusing DVL, IMU, and PS before training')
plt.legend()
plt.grid()
plt.show()

# Plot the estimated trajectory in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_x, positions_y, positions_z, label='Estimated Trajectory fusing DVL, IMU and PS', color='blue')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], color='green', label='Start (Ground Truth)', zorder=5)
ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], color='red', label='End (Ground Truth)', zorder=5)
ax.scatter(positions_x[0], positions_y[0], positions_z[0], color='purple', label='Start (Estimated)', zorder=5)
ax.scatter(positions_x[-1], positions_y[-1], positions_z[-1], color='brown', label='End (Estimated)', zorder=5)
ax.set_xlabel('X Position (m)') 
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Vehicle Trajectory Estimated by Fusing DVL, IMU, and PS before training')
ax.legend()
plt.show()


# Prepare the input data for the Neural Kalman Filter
estimated_position = np.column_stack((positions_x, positions_y, positions_z))
estimated_orientation = imu_orientations[:, :4]  # Use the first 4 columns for quaternion
ground_truth = np.column_stack((gt_positions, gt_orientations))  # Combine position and orientation

# Normalize the data
estimated_position = (estimated_position - np.mean(estimated_position, axis=0)) / np.std(estimated_position, axis=0)
estimated_orientation = (estimated_orientation - np.mean(estimated_orientation, axis=0)) / np.std(estimated_orientation, axis=0)
ground_truth = (ground_truth - np.mean(ground_truth, axis=0)) / np.std(ground_truth, axis=0)

print("Estimated Position Shape:", estimated_position.shape)
print("Estimated Orientation Shape:", estimated_orientation.shape)
print("Ground Truth Shape:", ground_truth.shape)
print("Estimated Position:", estimated_position[0])
print("Estimated Orientation:", estimated_orientation[0])
print("Ground Truth:", ground_truth[0])

inputs = np.hstack((estimated_position, estimated_orientation))
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(ground_truth, dtype=torch.float32)

# Define the Neural Kalman Filter model
class NeuralKalmanFilter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralKalmanFilter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
input_dim = inputs.shape[1]
hidden_dim = 64
output_dim = labels.shape[1]

# Initialize the model, loss function, and optimizer
model = NeuralKalmanFilter(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Correct method call
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
# torch.save(model.state_dict(), 'nkf_odoinput_001.pth')

# print("Training complete. Model saved as 'nkf_odoinput_001.pth'.")

# Load the trained model for evaluation
model.load_state_dict(torch.load('nkf_odoinput_001.pth'))
model.eval()

# Get the model predictions
with torch.no_grad():
    predictions = model(inputs).numpy()

# Plot the results
fig = plt.figure(figsize=(10, 10))

# Plot positions
ax1 = fig.add_subplot(211, projection='3d')
ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth')
ax1.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions')
ax1.set_title('3D Vehicle Trajectory after training')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Plot orientations (quaternions)
ax2 = fig.add_subplot(212)
ax2.plot(ground_truth[:, 3], label='Ground Truth W')
ax2.plot(ground_truth[:, 4], label='Ground Truth X')
ax2.plot(ground_truth[:, 5], label='Ground Truth Y')
ax2.plot(ground_truth[:, 6], label='Ground Truth Z')
ax2.plot(predictions[:, 3], label='Predictions W')
ax2.plot(predictions[:, 4], label='Predictions X')
ax2.plot(predictions[:, 5], label='Predictions Y')
ax2.plot(predictions[:, 6], label='Predictions Z')
ax2.set_title('Orientation (Quaternions)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Quaternion Value')
ax2.legend()

plt.tight_layout()
plt.show()
