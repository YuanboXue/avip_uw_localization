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
data = pd.read_csv('../data/synch_pool_mix.csv')

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

# Set the known start point (use the first ground truth position)
start_point = gt_positions[0].copy()

# Initialize variables for estimated positions
positions_x = [start_point[0]]
positions_y = [start_point[1]]
positions_z = [start_point[2]]
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
estimated_position = np.column_stack((positions_x, positions_y, positions_z))

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

# Calculate displacement vectors (from one step to the next)
# For estimated trajectory
displacement_est = np.zeros_like(estimated_position)
displacement_est[1:] = estimated_position[1:] - estimated_position[:-1]

# For ground truth
displacement_gt_pos = np.zeros_like(gt_positions)
displacement_gt_pos[1:] = gt_positions[1:] - gt_positions[:-1]

# Prepare the input data for the Neural Network
estimated_orientation = imu_orientations[:, :4]  # Use the first 4 columns for quaternion
displacement_gt = np.column_stack((displacement_gt_pos, gt_orientations))  # Combine displacement and orientation

# Split data into 80% training and 20% evaluation
split_idx = int(len(displacement_est) * 0.8)
train_displacement = displacement_est[:split_idx]
train_orientation = estimated_orientation[:split_idx]
train_gt_displacement = displacement_gt[:split_idx]

eval_displacement = displacement_est[split_idx:]
eval_orientation = estimated_orientation[split_idx:]
eval_gt_displacement = displacement_gt[split_idx:]

# Normalize the training data
train_displacement_mean = np.mean(train_displacement, axis=0)
train_displacement_std = np.std(train_displacement, axis=0)
train_displacement = (train_displacement - train_displacement_mean) / train_displacement_std

train_orientation_mean = np.mean(train_orientation, axis=0)
train_orientation_std = np.std(train_orientation, axis=0)
train_orientation = (train_orientation - train_orientation_mean) / train_orientation_std

train_gt_displacement_mean = np.mean(train_gt_displacement, axis=0)
train_gt_displacement_std = np.std(train_gt_displacement, axis=0)
train_gt_displacement = (train_gt_displacement - train_gt_displacement_mean) / train_gt_displacement_std

# Normalize the evaluation data using training statistics
eval_displacement = (eval_displacement - train_displacement_mean) / train_displacement_std
eval_orientation = (eval_orientation - train_orientation_mean) / train_orientation_std
eval_gt_displacement = (eval_gt_displacement - train_gt_displacement_mean) / train_gt_displacement_std

print("Training Data:")
print("Displacement Shape:", train_displacement.shape)
print("Orientation Shape:", train_orientation.shape)
print("Ground Truth Displacement Shape:", train_gt_displacement.shape)

print("\nEvaluation Data:")
print("Displacement Shape:", eval_displacement.shape)
print("Orientation Shape:", eval_orientation.shape)
print("Ground Truth Displacement Shape:", eval_gt_displacement.shape)

# Prepare inputs and labels for training
train_inputs = np.hstack((train_displacement, train_orientation))
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_gt_displacement, dtype=torch.float32)

# Prepare inputs and labels for evaluation
eval_inputs = np.hstack((eval_displacement, eval_orientation))
eval_inputs = torch.tensor(eval_inputs, dtype=torch.float32)
eval_labels = torch.tensor(eval_gt_displacement, dtype=torch.float32)

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
input_dim = train_inputs.shape[1]
hidden_dim = 64
output_dim = train_labels.shape[1]

# Initialize the model, loss function, and optimizer
model = NeuralKalmanFilter(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with evaluation
num_epochs = 100
train_losses = []
eval_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    train_outputs = model(train_inputs)
    train_loss = criterion(train_outputs, train_labels)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        eval_outputs = model(eval_inputs)
        eval_loss = criterion(eval_outputs, eval_labels)
        eval_losses.append(eval_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Eval Loss: {eval_loss.item():.4f}')

# Save the trained model
# torch.save(model.state_dict(), 'nkf_ododisp_0026_8020.pth')
# print("Training complete. Model saved as 'nkf_ododisp_0026_8020.pth'.")

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the evaluation set
# model.eval()
# model.load_state_dict(torch.load('nkf_ododisp_001_8020.pth'))
model.eval()
with torch.no_grad():
    eval_predictions = model(eval_inputs).numpy()

# Denormalize predictions and ground truth for evaluation set
eval_predictions_denorm = eval_predictions * train_gt_displacement_std + train_gt_displacement_mean
eval_gt_denorm = eval_gt_displacement * train_gt_displacement_std + train_gt_displacement_mean

# Calculate the cumulative trajectory by integrating the predicted displacements
# Start from the last point of the training set
start_idx = split_idx
start_position = estimated_position[start_idx]

predicted_trajectory = [start_position]
actual_trajectory = [gt_positions[start_idx]]

for i in range(len(eval_predictions_denorm)):
    # Add predicted displacement to get next position
    next_point = predicted_trajectory[-1] + eval_predictions_denorm[i, :3]
    predicted_trajectory.append(next_point)
    
    # Add actual displacement to get next ground truth position
    next_actual = actual_trajectory[-1] + eval_gt_denorm[i, :3]
    actual_trajectory.append(next_actual)

predicted_trajectory = np.array(predicted_trajectory)
actual_trajectory = np.array(actual_trajectory)

# Plot the predicted vs actual trajectory
plt.figure(figsize=(10, 6))
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='Predicted Trajectory', color='blue')
plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], color='green', label='Start', zorder=5)
plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], color='red', label='End (Ground Truth)', zorder=5)
plt.scatter(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], color='purple', label='End (Predicted)', zorder=5)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2], 
        label='Ground Truth', linestyle='--', color='orange')
ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], 
        label='Predicted Trajectory', color='blue')
ax.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], actual_trajectory[0, 2], 
           color='green', label='Start', zorder=5)
ax.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], actual_trajectory[-1, 2], 
           color='red', label='End (Ground Truth)', zorder=5)
ax.scatter(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], predicted_trajectory[-1, 2], 
           color='purple', label='End (Predicted)', zorder=5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
ax.legend()
plt.show()

# Calculate error metrics
position_errors = np.sqrt(np.sum((actual_trajectory - predicted_trajectory)**2, axis=1))

plt.figure(figsize=(10, 5))
plt.plot(position_errors)
plt.xlabel('Time Step')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)
plt.show()

print(f"Mean Position Error: {np.mean(position_errors):.4f} m")
print(f"Max Position Error: {np.max(position_errors):.4f} m")
print(f"Final Position Error: {position_errors[-1]:.4f} m")