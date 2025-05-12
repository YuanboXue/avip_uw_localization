# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
# from sklearn.metrics import mean_squared_error


# # Load the synchronized CSV file
# # 1) for each DVL timestamp, use interpolation to resample the IMU data; 
# # 2) for each DVL timestamp, find the nearest pressure sensor data; 
# # 3) for each DVL timestamp, find the nearest ground truth as the train label
# # Load the synchronized CSV files - one for training, one for evaluation
# train_data = pd.read_csv('../data/synch_pool_mix.csv')
# eval_data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# # Helper function to convert quaternion to rotation matrix
# def quaternion_to_rotation_matrix(q):
#     rotation = R.from_quat(q)
#     return rotation.as_matrix()

# # Function to process the data (extract features and calculate dead reckoning)
# def process_data(data):
#     # Extract the relevant columns
#     timestamps = data['timestamp'].values
#     dvl_velocities = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
#     imu_orientations = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w']].values
#     pressure_data = data['pressure_fluid_pressure'].values
#     gt_positions = data[['gt_position_x', 'gt_position_y', 'gt_position_z']].values
#     gt_orientations = data[['gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

#     # Calculate water depth from pressure sensor
#     water_depth = - (pressure_data * 1000 - 101325)/(1000*9.80665)

#     # Set the known start point (use the first ground truth position)
#     start_point = gt_positions[0].copy()

#     # Initialize variables for estimated positions
#     positions_x = [start_point[0]]
#     positions_y = [start_point[1]]
#     positions_z = [start_point[2]]
#     prev_time = timestamps[0]

#     # Dead reckoning fusing IMU, DVL, and Pressure Sensor
#     for i in range(1, len(timestamps)):
#         # Time difference (dt)
#         current_time = timestamps[i]
#         dt = (current_time - prev_time) / 1e9  # Convert nanoseconds to seconds

#         # Get DVL velocities
#         dvl_velocity = dvl_velocities[i]

#         # Get IMU orientation and convert to rotation matrix
#         imu_orientation = imu_orientations[i]
#         rotation_matrix = quaternion_to_rotation_matrix(imu_orientation)

#         # Rotate DVL velocities into the global frame
#         global_velocity = rotation_matrix @ dvl_velocity

#         # Update positions using velocity integration
#         velocity_x = positions_x[-1] + global_velocity[0] * dt
#         velocity_y = positions_y[-1] + global_velocity[1] * dt
#         velocity_z = water_depth[i]  # Replace z-position with water depth

#         positions_x.append(velocity_x)
#         positions_y.append(velocity_y)
#         positions_z.append(velocity_z)

#         # Update previous time
#         prev_time = current_time

#     # Convert positions to numpy arrays
#     positions_x = np.array(positions_x)
#     positions_y = np.array(positions_y)
#     positions_z = np.array(positions_z)
#     estimated_position = np.column_stack((positions_x, positions_y, positions_z))
    
#     # Calculate displacement vectors (from one step to the next)
#     # For estimated trajectory
#     displacement_est = np.zeros_like(estimated_position)
#     displacement_est[1:] = estimated_position[1:] - estimated_position[:-1]

#     # For ground truth
#     displacement_gt_pos = np.zeros_like(gt_positions)
#     displacement_gt_pos[1:] = gt_positions[1:] - gt_positions[:-1]
    
#     # Prepare the input data for the Neural Network
#     estimated_orientation = imu_orientations[:, :4]  # Use the first 4 columns for quaternion
#     displacement_gt = np.column_stack((displacement_gt_pos, gt_orientations))  # Combine displacement and orientation
    
#     return {
#         'timestamps': timestamps,
#         'estimated_position': estimated_position,
#         'gt_positions': gt_positions,
#         'displacement_est': displacement_est,
#         'displacement_gt': displacement_gt,
#         'estimated_orientation': estimated_orientation,
#         'gt_orientations': gt_orientations
#     }

# # Process training and evaluation data
# train_processed = process_data(train_data)
# eval_processed = process_data(eval_data)

# # Plot the estimated trajectory and ground truth for training data
# plt.figure(figsize=(10, 6))
# plt.plot(train_processed['estimated_position'][:, 0], train_processed['estimated_position'][:, 1], 
#          label='Estimated Trajectory fusing DVL, IMU and PS', color='blue')
# plt.plot(train_processed['gt_positions'][:, 0], train_processed['gt_positions'][:, 1], 
#          label='Ground Truth', linestyle='--', color='orange')
# plt.scatter(train_processed['gt_positions'][0, 0], train_processed['gt_positions'][0, 1], 
#            color='green', label='Start (Ground Truth)', zorder=5)
# plt.scatter(train_processed['gt_positions'][-1, 0], train_processed['gt_positions'][-1, 1], 
#            color='red', label='End (Ground Truth)', zorder=5)
# plt.scatter(train_processed['estimated_position'][0, 0], train_processed['estimated_position'][0, 1], 
#            color='purple', label='Start (Estimated)', zorder=5)
# plt.scatter(train_processed['estimated_position'][-1, 0], train_processed['estimated_position'][-1, 1], 
#            color='brown', label='End (Estimated)', zorder=5)
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('Vehicle Trajectory Estimated by Fusing DVL, IMU, and PS (Training Data)')
# plt.legend()
# plt.grid()
# plt.show()

# # Print the estimated z-positions and ground truth z-positions
# # print("Estimated Z-Positions (Training Data):", train_processed['estimated_position'][:, 2])
# # print("Ground Truth Z-Positions (Training Data):", train_processed['gt_positions'][:, 2])

# # Plot the estimated z-positions and ground truth z-positions
# # plt.figure(figsize=(10, 6))
# # plt.plot(train_processed['timestamps'], train_processed['estimated_position'][:, 2],
# #             label='Estimated Z-Position', color='blue')
# # plt.plot(train_processed['timestamps'], train_processed['gt_positions'][:, 2],
# #             label='Ground Truth Z-Position', linestyle='--', color='orange')
# # plt.xlabel('Timestamp')
# # plt.ylabel('Z Position (m)')
# # plt.title('Z Position Over Time (Training Data)')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # Normalize the training data
# train_displacement = train_processed['displacement_est']
# train_orientation = train_processed['estimated_orientation']
# train_gt_displacement = train_processed['displacement_gt']

# train_displacement_mean = np.mean(train_displacement, axis=0)
# train_displacement_std = np.std(train_displacement, axis=0)
# train_displacement = (train_displacement - train_displacement_mean) / train_displacement_std

# train_orientation_mean = np.mean(train_orientation, axis=0)
# train_orientation_std = np.std(train_orientation, axis=0)
# train_orientation = (train_orientation - train_orientation_mean) / train_orientation_std

# train_gt_displacement_mean = np.mean(train_gt_displacement, axis=0)
# train_gt_displacement_std = np.std(train_gt_displacement, axis=0)
# train_gt_displacement = (train_gt_displacement - train_gt_displacement_mean) / train_gt_displacement_std

# # Normalize the evaluation data using training statistics
# eval_displacement = eval_processed['displacement_est']
# eval_orientation = eval_processed['estimated_orientation']
# eval_gt_displacement = eval_processed['displacement_gt']

# eval_displacement = (eval_displacement - train_displacement_mean) / train_displacement_std
# eval_orientation = (eval_orientation - train_orientation_mean) / train_orientation_std
# eval_gt_displacement = (eval_gt_displacement - train_gt_displacement_mean) / train_gt_displacement_std

# print("Training Data:")
# print("Displacement Shape:", train_displacement.shape)
# print("Orientation Shape:", train_orientation.shape)
# print("Ground Truth Displacement Shape:", train_gt_displacement.shape)

# print("\nEvaluation Data:")
# print("Displacement Shape:", eval_displacement.shape)
# print("Orientation Shape:", eval_orientation.shape)
# print("Ground Truth Displacement Shape:", eval_gt_displacement.shape)

# # Prepare inputs and labels for training
# train_inputs = np.hstack((train_displacement, train_orientation))
# train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
# train_labels = torch.tensor(train_gt_displacement, dtype=torch.float32)

# # Prepare inputs and labels for evaluation
# eval_inputs = np.hstack((eval_displacement, eval_orientation))
# eval_inputs = torch.tensor(eval_inputs, dtype=torch.float32)
# eval_labels = torch.tensor(eval_gt_displacement, dtype=torch.float32)

# # Define the Neural Network model
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # class NeuralNetwork(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(NeuralNetwork, self).__init__()
# #         self.fc1 = nn.Linear(input_dim, 128)
# #         self.fc2 = nn.Linear(128, 64)
# #         self.fc3 = nn.Linear(64, 32)
# #         self.fc4 = nn.Linear(32, 16)
# #         self.fc5 = nn.Linear(16, output_dim)
# #         # self.fc4 = nn.Linear(32, output_dim)
    
# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         x = torch.relu(self.fc3(x))
# #         x = torch.relu(self.fc4(x))
# #         x = self.fc5(x)
# #         return x

# # Hyperparameters
# input_dim = train_inputs.shape[1]
# hidden_dim = 128
# output_dim = train_labels.shape[1]

# # Initialize the model, loss function, and optimizer
# model = NeuralNetwork(input_dim, hidden_dim, output_dim)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop with evaluation
# num_epochs = 900
# train_losses = []
# eval_losses = []

# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     optimizer.zero_grad()
#     train_outputs = model(train_inputs)
#     train_loss = criterion(train_outputs, train_labels)
#     train_loss.backward()
#     optimizer.step()
#     train_losses.append(train_loss.item())
    
#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         eval_outputs = model(eval_inputs)
#         eval_loss = criterion(eval_outputs, eval_labels)
#         eval_losses.append(eval_loss.item())
    
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Eval Loss: {eval_loss.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'nkf_ododisp_mix.pth')
# print("Training complete. Model saved as 'nkf_ododisp_mix.pth'.")

# # Plot the loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(eval_losses, label='Evaluation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Evaluation Losses')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Evaluate the model on the evaluation set
# # model.load_state_dict(torch.load('nkf_ododisp_mix.pth'))
# model.eval()
# with torch.no_grad():
#     eval_predictions = model(eval_inputs).numpy()

# # Denormalize predictions and ground truth for evaluation set
# eval_predictions_denorm = eval_predictions * train_gt_displacement_std + train_gt_displacement_mean
# eval_gt_denorm = eval_gt_displacement * train_gt_displacement_std + train_gt_displacement_mean

# # Calculate the cumulative trajectory by integrating the predicted displacements
# # Start from the first point of the evaluation set
# start_position = eval_processed['gt_positions'][0]  # Start from ground truth position

# predicted_trajectory = [start_position]
# actual_trajectory = [eval_processed['gt_positions'][0]]

# for i in range(len(eval_predictions_denorm)):
#     # Add predicted displacement to get next position
#     next_point = predicted_trajectory[-1] + eval_predictions_denorm[i, :3]
#     predicted_trajectory.append(next_point)
    
#     # Add actual displacement to get next ground truth position
#     next_actual = actual_trajectory[-1] + eval_gt_denorm[i, :3]
#     actual_trajectory.append(next_actual)

# predicted_trajectory = np.array(predicted_trajectory)
# actual_trajectory = np.array(actual_trajectory)

# # Plot the predicted vs actual trajectory
# plt.figure(figsize=(10, 6))
# plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], label='Ground Truth', linestyle='--', color='orange')
# plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='Predicted Trajectory', color='blue')
# plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], color='green', label='Start', zorder=5)
# plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], color='red', label='End (Ground Truth)', zorder=5)
# plt.scatter(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], color='purple', label='End (Predicted)', zorder=5)
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('2D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot the 3D trajectory
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2], 
#         label='Ground Truth', linestyle='--', color='orange')
# ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], 
#         label='Predicted Trajectory', color='blue')
# ax.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], actual_trajectory[0, 2], 
#            color='green', label='Start', zorder=5)
# ax.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], actual_trajectory[-1, 2], 
#            color='red', label='End (Ground Truth)', zorder=5)
# ax.scatter(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], predicted_trajectory[-1, 2], 
#            color='purple', label='End (Predicted)', zorder=5)
# ax.set_xlabel('X Position (m)')
# ax.set_ylabel('Y Position (m)')
# ax.set_zlabel('Z Position (m)')
# ax.set_title('3D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
# ax.legend()
# plt.show()

# # Calculate error metrics
# position_errors = np.sqrt(np.sum((actual_trajectory - predicted_trajectory)**2, axis=1))

# plt.figure(figsize=(10, 5))
# plt.plot(position_errors)
# plt.xlabel('Time Step')
# plt.ylabel('Position Error (m)')
# plt.title('Position Error Over Time')
# plt.grid(True)
# plt.show()

# # Calculate the root mean square error of absolute error
# mse = mean_squared_error(actual_trajectory, predicted_trajectory)
# rmse_ = (mse) ** (1/2)
# print(f"RMSE of Position Error from sklearn: {rmse_:.4f} m")

# meanSquaredError = ((actual_trajectory - predicted_trajectory) ** 2).mean()
# rmse__ = np.sqrt(meanSquaredError)
# print(f"RMSE of Position Error from numpy: {rmse__:.4f} m")

# # Evaluating orientation predictions
# # Extract orientation quaternions from denormalized predictions and ground truth
# predicted_orientations = eval_predictions_denorm[:, 3:7]  # Extract orientation quaternions from predictions
# actual_orientations = eval_gt_denorm[:, 3:7]  # Extract orientation quaternions from ground truth

# # 2. Calculate Roll, Pitch, Yaw errors
# def quaternion_to_euler(q):
#     """Convert quaternion to Euler angles (in degrees)"""
#     r = R.from_quat(q)
#     return r.as_euler('xyz', degrees=True)  # xyz order (roll, pitch, yaw)

# # Convert quaternions to Euler angles
# predicted_euler = np.array([quaternion_to_euler(q) for q in predicted_orientations])
# actual_euler = np.array([quaternion_to_euler(q) for q in actual_orientations])

# # Calculate errors for each Euler angle
# roll_error = np.abs(predicted_euler[:, 0] - actual_euler[:, 0])
# pitch_error = np.abs(predicted_euler[:, 1] - actual_euler[:, 1])
# yaw_error = np.abs(predicted_euler[:, 2] - actual_euler[:, 2])

# # Account for angle wrapping (e.g., 359° vs 1° should be 2° difference, not 358°)
# roll_error = np.minimum(roll_error, 360 - roll_error)
# pitch_error = np.minimum(pitch_error, 360 - pitch_error)
# yaw_error = np.minimum(yaw_error, 360 - yaw_error)

# # Plot Euler angle errors
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(roll_error)
# plt.ylabel('Roll Error (degrees)')
# plt.title('Roll Error Over Time')
# plt.grid(True)

# plt.subplot(3, 1, 2)
# plt.plot(pitch_error)
# plt.ylabel('Pitch Error (degrees)')
# plt.title('Pitch Error Over Time')
# plt.grid(True)

# plt.subplot(3, 1, 3)
# plt.plot(yaw_error)
# plt.xlabel('Time Step')
# plt.ylabel('Yaw Error (degrees)')
# plt.title('Yaw Error Over Time')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Print statistics for Euler angle errors
# print(f"Mean Roll Error: {np.mean(roll_error):.4f} degrees")
# print(f"Mean Pitch Error: {np.mean(pitch_error):.4f} degrees")
# print(f"Mean Yaw Error: {np.mean(yaw_error):.4f} degrees")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error

# Load the synchronized CSV files - one for training, one for evaluation
train_data = pd.read_csv('../data/synch_pool_mix.csv')
eval_data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Helper function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    rotation = R.from_quat(q)
    return rotation.as_matrix()

# Function to process the data (extract features and calculate dead reckoning)
def process_data(data):
    # Extract the relevant columns
    timestamps = data['timestamp'].values
    dvl_velocities = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
    imu_orientations = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w']].values
    pressure_data = data['pressure_fluid_pressure'].values
    gt_positions = data[['gt_position_x', 'gt_position_y', 'gt_position_z']].values
    gt_orientations = data[['gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

    # Calculate water depth from pressure sensor
    water_depth = - (pressure_data * 1000 - 101325)/(1000*9.80665)

    # Prepare the input data for the Neural Network
    sensor_inputs = np.hstack((dvl_velocities, imu_orientations, water_depth.reshape(-1, 1)))
    gt_pose = np.hstack((gt_positions, gt_orientations))  # Combine position and orientation

    return {
        'timestamps': timestamps,
        'sensor_inputs': sensor_inputs,
        'gt_pose': gt_pose
    }

# Process training and evaluation data
train_processed = process_data(train_data)
eval_processed = process_data(eval_data)

# Normalize the training data
train_inputs = train_processed['sensor_inputs']
train_gt_pose = train_processed['gt_pose']

train_inputs_mean = np.mean(train_inputs, axis=0)
train_inputs_std = np.std(train_inputs, axis=0)
train_inputs = (train_inputs - train_inputs_mean) / train_inputs_std

train_gt_pose_mean = np.mean(train_gt_pose, axis=0)
train_gt_pose_std = np.std(train_gt_pose, axis=0)
train_gt_pose = (train_gt_pose - train_gt_pose_mean) / train_gt_pose_std

# Normalize the evaluation data using training statistics
eval_inputs = eval_processed['sensor_inputs']
eval_gt_pose = eval_processed['gt_pose']

eval_inputs = (eval_inputs - train_inputs_mean) / train_inputs_std
eval_gt_pose = (eval_gt_pose - train_gt_pose_mean) / train_gt_pose_std

print("Training Data:")
print("Inputs Shape:", train_inputs.shape)
print("Ground Truth Pose Shape:", train_gt_pose.shape)

print("\nEvaluation Data:")
print("Inputs Shape:", eval_inputs.shape)
print("Ground Truth Pose Shape:", eval_gt_pose.shape)

# Prepare inputs and labels for training
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_gt_pose, dtype=torch.float32)

# Prepare inputs and labels for evaluation
eval_inputs = torch.tensor(eval_inputs, dtype=torch.float32)
eval_labels = torch.tensor(eval_gt_pose, dtype=torch.float32)

# Define the Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
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
hidden_dim = 128
output_dim = train_labels.shape[1]

# Initialize the model, loss function, and optimizer
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with evaluation
num_epochs = 900
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
# torch.save(model.state_dict(), 'sensor_based_localization.pth')
# print("Training complete. Model saved as 'sensor_based_localization.pth'.")

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
model.eval()
with torch.no_grad():
    eval_predictions = model(eval_inputs).numpy()

# Denormalize predictions and ground truth for evaluation set
eval_predictions_denorm = eval_predictions * train_gt_pose_std + train_gt_pose_mean
eval_gt_denorm = eval_gt_pose * train_gt_pose_std + train_gt_pose_mean

# Plot the predicted vs actual trajectory
plt.figure(figsize=(10, 6))
plt.plot(eval_gt_denorm[:, 0], eval_gt_denorm[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.plot(eval_predictions_denorm[:, 0], eval_predictions_denorm[:, 1], label='Predicted Trajectory', color='blue')
plt.scatter(eval_gt_denorm[0, 0], eval_gt_denorm[0, 1], color='green', label='Start', zorder=5)
plt.scatter(eval_gt_denorm[-1, 0], eval_gt_denorm[-1, 1], color='red', label='End (Ground Truth)', zorder=5)
plt.scatter(eval_predictions_denorm[-1, 0], eval_predictions_denorm[-1, 1], color='purple', label='End (Predicted)', zorder=5)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(eval_gt_denorm[:, 0], eval_gt_denorm[:, 1], eval_gt_denorm[:, 2],
        label='Ground Truth', linestyle='--', color='orange')
ax.plot(eval_predictions_denorm[:, 0], eval_predictions_denorm[:, 1], eval_predictions_denorm[:, 2],
        label='Predicted Trajectory', color='blue')
ax.scatter(eval_gt_denorm[0, 0], eval_gt_denorm[0, 1], eval_gt_denorm[0, 2],
           color='green', label='Start', zorder=5)
ax.scatter(eval_gt_denorm[-1, 0], eval_gt_denorm[-1, 1], eval_gt_denorm[-1, 2],
           color='red', label='End (Ground Truth)', zorder=5)
ax.scatter(eval_predictions_denorm[-1, 0], eval_predictions_denorm[-1, 1], eval_predictions_denorm[-1, 2],
           color='purple', label='End (Predicted)', zorder=5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory: Ground Truth vs Predicted (Evaluation Set)')
ax.legend()
plt.show()

def moving_average(data, window_size=5):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

# Smooth the predicted trajectory
window_size = 25  # Adjust the window size as needed
smoothed_predictions_x = moving_average(eval_predictions_denorm[:, 0], window_size)
smoothed_predictions_y = moving_average(eval_predictions_denorm[:, 1], window_size)
smoothed_predictions_z = moving_average(eval_predictions_denorm[:, 2], window_size)

# Adjust ground truth to match the smoothed prediction length
smoothed_ground_truth_x = eval_gt_denorm[window_size-1:, 0]
smoothed_ground_truth_y = eval_gt_denorm[window_size-1:, 1]
smoothed_ground_truth_z = eval_gt_denorm[window_size-1:, 2]

# Plot the smoothed trajectory
plt.figure(figsize=(10, 6))
plt.plot(smoothed_ground_truth_x, smoothed_ground_truth_y, label='Ground Truth (Smoothed)', linestyle='--', color='orange')
plt.plot(smoothed_predictions_x, smoothed_predictions_y, label='Predictions (Smoothed)', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Smoothed 2D Trajectory: Ground Truth vs Predictions')
plt.legend()
plt.grid()
plt.show()

# Calculate error metrics
position_errors = np.sqrt(np.sum((eval_gt_denorm[:, :3] - eval_predictions_denorm[:, :3])**2, axis=1))

plt.figure(figsize=(10, 5))
plt.plot(position_errors)
plt.xlabel('Time Step')
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)
plt.show()

# Calculate the root mean square error of absolute error
mse = mean_squared_error(eval_gt_denorm[:, :3], eval_predictions_denorm[:, :3])
rmse_ = (mse) ** (1/2)
print(f"RMSE of Position Error from sklearn: {rmse_:.4f} m")

meanSquaredError = ((eval_gt_denorm[:, :3] - eval_predictions_denorm[:, :3]) ** 2).mean()
rmse__ = np.sqrt(meanSquaredError)
print(f"RMSE of Position Error from numpy: {rmse__:.4f} m")

# Evaluating orientation predictions
# Extract orientation quaternions from denormalized predictions and ground truth
predicted_orientations = eval_predictions_denorm[:, 3:7] # Extract orientation quaternions from predictions
actual_orientations = eval_gt_denorm[:, 3:7] # Extract orientation quaternions from ground truth

# 2. Calculate Roll, Pitch, Yaw errors
def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (in degrees)"""
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=True) # xyz order (roll, pitch, yaw)

# Convert quaternions to Euler angles
predicted_euler = np.array([quaternion_to_euler(q) for q in predicted_orientations])
actual_euler = np.array([quaternion_to_euler(q) for q in actual_orientations])

# Calculate errors for each Euler angle
roll_error = np.abs(predicted_euler[:, 0] - actual_euler[:, 0])
pitch_error = np.abs(predicted_euler[:, 1] - actual_euler[:, 1])
yaw_error = np.abs(predicted_euler[:, 2] - actual_euler[:, 2])

# Account for angle wrapping (e.g., 359° vs 1° should be 2° difference, not 358°)
roll_error = np.minimum(roll_error, 360 - roll_error)
pitch_error = np.minimum(pitch_error, 360 - pitch_error)
yaw_error = np.minimum(yaw_error, 360 - yaw_error)

# Plot Euler angle errors
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(roll_error)
plt.ylabel('Roll Error (degrees)')
plt.title('Roll Error Over Time')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(pitch_error)
plt.ylabel('Pitch Error (degrees)')
plt.title('Pitch Error Over Time')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(yaw_error)
plt.xlabel('Time Step')
plt.ylabel('Yaw Error (degrees)')
plt.title('Yaw Error Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print statistics for Euler angle errors
print(f"Mean Roll Error: {np.mean(roll_error):.4f} degrees")
print(f"Mean Pitch Error: {np.mean(pitch_error):.4f} degrees")
print(f"Mean Yaw Error: {np.mean(yaw_error):.4f} degrees")