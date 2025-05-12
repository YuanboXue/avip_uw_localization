import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

# Load the synchronized CSV files
train_data = pd.read_csv('../data/synch_pool_mix_check.csv')
eval_data = pd.read_csv('../data/synch_pool_shr_train01.csv')

print(f"Loaded training dataset with {len(train_data)} samples")
print(f"Loaded evaluation dataset with {len(eval_data)} samples")

# Function to extract features from dataset
def extract_features(data):
    timestamps = data['timestamp'].values
    imu_data = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w',
                    'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
                    'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z']].values
    dvl_data = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
    pressure_data = data[['pressure_fluid_pressure']].values
    ground_truth = data[['gt_position_x', 'gt_position_y', 'gt_position_z',
                        'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values
    
    return timestamps, imu_data, dvl_data, pressure_data, ground_truth

# Extract features from training and evaluation datasets
train_timestamps, train_imu_data, train_dvl_data, train_pressure_data, train_ground_truth = extract_features(train_data)
eval_timestamps, eval_imu_data, eval_dvl_data, eval_pressure_data, eval_ground_truth = extract_features(eval_data)

# Check for NaN values in the input data
print("\nTRAINING DATA:")
print("NaN values in IMU data:", np.isnan(train_imu_data).any())
print("NaN values in DVL data:", np.isnan(train_dvl_data).any())
print("NaN values in Pressure data:", np.isnan(train_pressure_data).any())
print("NaN values in Ground Truth data:", np.isnan(train_ground_truth).any())

print("\nEVALUATION DATA:")
print("NaN values in IMU data:", np.isnan(eval_imu_data).any())
print("NaN values in DVL data:", np.isnan(eval_dvl_data).any())
print("NaN values in Pressure data:", np.isnan(eval_pressure_data).any())
print("NaN values in Ground Truth data:", np.isnan(eval_ground_truth).any())

# Define the IMU Encoder
class IMUEncoder(nn.Module):
    def __init__(self, input_dim=10, output_dim=64):
        super(IMUEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return x

# Define the DVL Encoder
class DVLEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(DVLEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return x

# Define the Pressure Encoder
class PressureEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=64):
        super(PressureEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return x

# Define the Neural Kalman Filter
class NeuralKalmanFilter(nn.Module):
    def __init__(self, imu_encoder, dvl_encoder, pressure_encoder, hidden_dim, output_dim):
        super(NeuralKalmanFilter, self).__init__()
        self.imu_encoder = imu_encoder
        self.dvl_encoder = dvl_encoder
        self.pressure_encoder = pressure_encoder

        # Fusion layer
        self.fc1 = nn.Linear(64 * 3, hidden_dim)  # 64 (IMU) + 64 (DVL) + 64 (Pressure)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.fc2 = nn.Linear(hidden_dim, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)  # Output: position (3) + orientation (4)

    def forward(self, imu_data, dvl_data, pressure_data):
        imu_features = self.imu_encoder(imu_data)
        dvl_features = self.dvl_encoder(dvl_data)
        pressure_features = self.pressure_encoder(pressure_data)

        # Concatenate features
        fused_features = torch.cat((imu_features, dvl_features, pressure_features), dim=1)

        # Pass through the fusion layers
        # x = torch.relu(self.fc1(fused_features))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

        x = torch.relu(self.fc1(fused_features))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Normalize the data using training statistics
def normalize_data(train_data, eval_data=None):
    # Calculate statistics from training data
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_std[train_std < 1e-8] = 1.0  # Avoid division by zero
    
    # Normalize the training data
    train_data_norm = (train_data - train_mean) / train_std
    
    # Normalize the evaluation data if provided
    if eval_data is not None:
        eval_data_norm = (eval_data - train_mean) / train_std
        return train_data_norm, eval_data_norm, train_mean, train_std
    else:
        return train_data_norm, train_mean, train_std

# Normalize all data types using training statistics
train_imu_norm, eval_imu_norm, imu_mean, imu_std = normalize_data(train_imu_data, eval_imu_data)
train_dvl_norm, eval_dvl_norm, dvl_mean, dvl_std = normalize_data(train_dvl_data, eval_dvl_data)
train_pressure_norm, eval_pressure_norm, pressure_mean, pressure_std = normalize_data(train_pressure_data, eval_pressure_data)
train_gt_norm, eval_gt_norm, gt_mean, gt_std = normalize_data(train_ground_truth, eval_ground_truth)

# Convert normalized data to PyTorch tensors
train_imu_tensor = torch.tensor(train_imu_norm, dtype=torch.float32)
train_dvl_tensor = torch.tensor(train_dvl_norm, dtype=torch.float32)
train_pressure_tensor = torch.tensor(train_pressure_norm, dtype=torch.float32)
train_gt_tensor = torch.tensor(train_gt_norm, dtype=torch.float32)

eval_imu_tensor = torch.tensor(eval_imu_norm, dtype=torch.float32)
eval_dvl_tensor = torch.tensor(eval_dvl_norm, dtype=torch.float32)
eval_pressure_tensor = torch.tensor(eval_pressure_norm, dtype=torch.float32)
eval_gt_tensor = torch.tensor(eval_gt_norm, dtype=torch.float32)

# Initialize the encoders and NKF
imu_encoder = IMUEncoder(input_dim=10, output_dim=64)
dvl_encoder = DVLEncoder(input_dim=3, output_dim=64)
pressure_encoder = PressureEncoder(input_dim=1, output_dim=64)
nkf = NeuralKalmanFilter(imu_encoder, dvl_encoder, pressure_encoder, hidden_dim=512, output_dim=7)

from torch.utils.data import TensorDataset, DataLoader

# Define batch size
batch_size = 64

# Create training dataset and dataloader
train_dataset = TensorDataset(train_imu_tensor, train_dvl_tensor, train_pressure_tensor, train_gt_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(nkf.parameters(), lr=0.001)

# Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training hyperparameters
# num_epochs = 1000
# train_losses = []
# eval_losses = []
# best_eval_loss = float('inf')
# patience = 10
# counter = 0

# print("\nStarting training...")
# Training loop
# for epoch in range(num_epochs):
#     # Set model to training mode
#     nkf.train()
    
#     # Train for one epoch
#     epoch_loss = 0
#     for batch_idx, (batch_imu, batch_dvl, batch_pressure, batch_gt) in enumerate(train_loader):
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = nkf(batch_imu, batch_dvl, batch_pressure)
#         loss = criterion(outputs, batch_gt)
        
#         # Check for NaN in loss
#         if torch.isnan(loss):
#             print(f"NaN detected in loss at epoch {epoch+1}, batch {batch_idx+1}")
#             print("Skipping this batch")
#             continue
            
#         # Backward pass and optimize
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(nkf.parameters(), max_norm=1.0)  # Gradient clipping
#         optimizer.step()
        
#         epoch_loss += loss.item()
    
#     # Calculate average epoch loss
#     avg_epoch_loss = epoch_loss / len(train_loader)
#     train_losses.append(avg_epoch_loss)
    
#     # Evaluation
#     nkf.eval()
#     with torch.no_grad():
#         eval_outputs = nkf(eval_imu_tensor, eval_dvl_tensor, eval_pressure_tensor)
#         eval_loss = criterion(eval_outputs, eval_gt_tensor).item()
#         eval_losses.append(eval_loss)
    
#     # Learning rate scheduler step
#     scheduler.step(eval_loss)
    
#     # Early stopping check
#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         # Save the best model
#         torch.save(nkf.state_dict(), 'nn_rawinput_mix.pth')
#         print(f"Epoch {epoch+1}/{num_epochs}: New best model saved (eval_loss: {eval_loss:.6f})")
#         counter = 0
#     else:
#         counter += 1
    
#     # Print epoch statistics
#     if (epoch + 1) % 5 == 0 or epoch == 0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Eval Loss: {eval_loss:.6f}")
    
#     # Early stopping
#     if counter >= patience:
#         print(f"Early stopping triggered after {epoch+1} epochs")
#         break

# Plot the training and evaluation loss
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(eval_losses, label='Evaluation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Evaluation Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# print("Training complete. Final model saved as 'nn_rawinput_mix.pth'")
# print(f"Best evaluation loss: {best_eval_loss:.6f}")

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = nkf(train_imu_tensor, train_dvl_tensor, train_pressure_tensor)
    loss = criterion(outputs, train_gt_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
# torch.save(nkf.state_dict(), 'nn_rawinput_mix.pth')
# print("Training complete. Model saved as 'nn_rawinput_mix.pth'.")

# Load the trained model weights
# try:
#     nkf.load_state_dict(torch.load('nn_odoinput_mix.pth'))
#     print("Successfully loaded model weights from nn_odoinput_mix.pth")
# except Exception as e:
#     print(f"Error loading model weights: {e}")
#     print("Using newly initialized model weights instead.")

# Set model to evaluation mode
nkf.eval()

# Evaluate on the evaluation dataset
with torch.no_grad():
    eval_predictions = nkf(eval_imu_tensor, eval_dvl_tensor, eval_pressure_tensor).numpy()

# Calculate loss
eval_loss = ((eval_predictions - eval_gt_norm)**2).mean()
print(f"Evaluation MSE loss: {eval_loss:.6f}")

# Denormalize predictions and ground truth
eval_predictions_denorm = eval_predictions * gt_std + gt_mean
eval_gt_denorm = eval_gt_norm * gt_std + gt_mean

# Plot the evaluation results
plt.figure(figsize=(10, 6))
plt.plot(eval_gt_denorm[:, 0], eval_gt_denorm[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.plot(eval_predictions_denorm[:, 0], eval_predictions_denorm[:, 1], label='Predictions', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Trajectory: Ground Truth vs Predictions (Evaluation Dataset)')
plt.legend()
plt.grid(True)
plt.show()

# Apply moving average to smooth the predicted trajectory
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

# Plot the smoothed 2D trajectory
plt.figure(figsize=(10, 6))
plt.plot(smoothed_ground_truth_x, smoothed_ground_truth_y, label='Ground Truth', linestyle='--', color='orange')
plt.plot(smoothed_predictions_x, smoothed_predictions_y, label='Predictions (Smoothed)', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Smoothed 2D Trajectory: Ground Truth vs Predictions (Evaluation Dataset)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the smoothed 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(smoothed_ground_truth_x, smoothed_ground_truth_y, smoothed_ground_truth_z, label='Ground Truth', linestyle='--', color='orange')
ax.plot(smoothed_predictions_x, smoothed_predictions_y, smoothed_predictions_z, label='Predictions (Smoothed)', color='blue')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Smoothed 3D Trajectory: Ground Truth vs Predictions (Evaluation Dataset)')
ax.legend()
plt.show()

# Calculate position error metrics
# position_errors = np.sqrt(np.sum((smoothed_ground_truth_x.reshape(-1, 1) - smoothed_predictions_x.reshape(-1, 1))**2 +
#                                 (smoothed_ground_truth_y.reshape(-1, 1) - smoothed_predictions_y.reshape(-1, 1))**2 +
#                                 (smoothed_ground_truth_z.reshape(-1, 1) - smoothed_predictions_z.reshape(-1, 1))**2, axis=1))

# Plot position errors
# plt.figure(figsize=(10, 5))
# plt.plot(position_errors)
# plt.xlabel('Time Step')
# plt.ylabel('Position Error (m)')
# plt.title('Position Error Over Time (Evaluation Dataset)')
# plt.grid(True)
# plt.show()

# print(f"Mean Position Error: {np.mean(position_errors):.4f} m")
# print(f"Max Position Error: {np.max(position_errors):.4f} m")
# print(f"RMSE: {np.sqrt(np.mean(position_errors**2)):.4f} m")

# Evaluate orientation predictions
# def quaternion_to_euler(q):
#     """Convert quaternion to Euler angles (in degrees)"""
#     r = R.from_quat(q)
#     return r.as_euler('xyz', degrees=True)  # xyz order (roll, pitch, yaw)

# def quaternion_angular_distance(q1, q2):
#     """Calculate angular distance between two quaternions in degrees"""
#     # Convert to rotation objects
#     r1 = R.from_quat(q1)
#     r2 = R.from_quat(q2)
    
#     # Calculate the relative rotation
#     r_diff = r1.inv() * r2
    
#     # Extract the angle from the relative rotation
#     angle = r_diff.magnitude() * (180.0 / np.pi)  # Convert to degrees
#     return angle

# Extract orientation quaternions from denormalized predictions and ground truth
# predicted_orientations = eval_predictions_denorm[:, 3:7]  # Extract orientation quaternions from predictions
# actual_orientations = eval_gt_denorm[:, 3:7]  # Extract orientation quaternions from ground truth

# Calculate orientation errors (angle between predicted and actual orientations)
# orientation_errors = np.array([quaternion_angular_distance(predicted_orientations[i], 
#                                                         actual_orientations[i])
#                             for i in range(len(predicted_orientations))])

# Plot orientation errors
# plt.figure(figsize=(10, 5))
# plt.plot(orientation_errors)
# plt.xlabel('Time Step')
# plt.ylabel('Orientation Error (degrees)')
# plt.title('Orientation Error Over Time')
# plt.grid(True)
# plt.show()

# print(f"Mean Orientation Error: {np.mean(orientation_errors):.4f} degrees")
# print(f"Max Orientation Error: {np.max(orientation_errors):.4f} degrees")
# print(f"RMSE of Orientation Error: {np.sqrt(np.mean(orientation_errors**2)):.4f} degrees")
