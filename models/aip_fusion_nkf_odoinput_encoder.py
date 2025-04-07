import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load the synchronized CSV file
data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Extract relevant columns
timestamps = data['timestamp'].values
imu_data = data[['imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w',
                 'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
                 'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z']].values
dvl_data = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
pressure_data = data[['pressure_fluid_pressure']].values
ground_truth = data[['gt_position_x', 'gt_position_y', 'gt_position_z',
                     'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, imu_data, dvl_data, pressure_data):
        imu_features = self.imu_encoder(imu_data)
        dvl_features = self.dvl_encoder(dvl_data)
        pressure_features = self.pressure_encoder(pressure_data)

        # Concatenate features
        fused_features = torch.cat((imu_features, dvl_features, pressure_features), dim=1)

        # Pass through the fusion layers
        x = torch.relu(self.fc1(fused_features))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Normalize the data
imu_data_mean = np.mean(imu_data, axis=0)
imu_data_std = np.std(imu_data, axis=0)
imu_data = (imu_data - imu_data_mean) / imu_data_std

dvl_data_mean = np.mean(dvl_data, axis=0)
dvl_data_std = np.std(dvl_data, axis=0)
dvl_data = (dvl_data - dvl_data_mean) / dvl_data_std

pressure_data_mean = np.mean(pressure_data, axis=0)
pressure_data_std = np.std(pressure_data, axis=0)
pressure_data = (pressure_data - pressure_data_mean) / pressure_data_std

ground_truth_mean = np.mean(ground_truth, axis=0)
ground_truth_std = np.std(ground_truth, axis=0)
ground_truth = (ground_truth - ground_truth_mean) / ground_truth_std

# Convert data to PyTorch tensors
imu_data = torch.tensor(imu_data, dtype=torch.float32)
dvl_data = torch.tensor(dvl_data, dtype=torch.float32)
pressure_data = torch.tensor(pressure_data, dtype=torch.float32)
ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

# Initialize the encoders and NKF
imu_encoder = IMUEncoder(input_dim=10, output_dim=64)
dvl_encoder = DVLEncoder(input_dim=3, output_dim=64)
pressure_encoder = PressureEncoder(input_dim=1, output_dim=64)
nkf = NeuralKalmanFilter(imu_encoder, dvl_encoder, pressure_encoder, hidden_dim=128, output_dim=7)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(nkf.parameters(), lr=0.001)

# Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()

#     # Forward pass
#     outputs = nkf(imu_data, dvl_data, pressure_data)
#     loss = criterion(outputs, ground_truth)

#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
# torch.save(nkf.state_dict(), 'nkf_single_timestep.pth')
# print("Training complete. Model saved as 'nkf_single_timestep.pth'.")

# Evaluate the model
nkf.load_state_dict(torch.load('nkf_single_timestep.pth'))
nkf.eval()
with torch.no_grad():
    predictions = nkf(imu_data, dvl_data, pressure_data).numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.plot(predictions[:, 0], predictions[:, 1], label='Predictions', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Trajectory: Ground Truth vs Predictions')
plt.legend()
plt.grid()
plt.show()

# Traj Smoothing
# Option 1: Apply Savitzky-Golay filter to smooth the trajectory
# window_length = 25  # Must be odd and greater than the polynomial order
# poly_order = 3
# smoothed_predictions_x = savgol_filter(predictions[:, 0], window_length, poly_order)
# smoothed_predictions_y = savgol_filter(predictions[:, 1], window_length, poly_order)
# smoothed_predictions_z = savgol_filter(predictions[:, 2], window_length, poly_order)

# Plot the smoothed trajectory
# plt.figure(figsize=(10, 6))
# plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', linestyle='--', color='orange')
# plt.plot(smoothed_predictions_x, smoothed_predictions_y, label='Predictions (Smoothed)', color='blue')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('Smoothed 2D Trajectory: Ground Truth vs Predictions')
# plt.legend()
# plt.grid()
# plt.show()

# Option 2: Apply moving average to smooth the trajectory
def moving_average(data, window_size=5):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

# Smooth the predicted trajectory
window_size = 25  # Adjust the window size as needed
smoothed_predictions_x = moving_average(predictions[:, 0], window_size)
smoothed_predictions_y = moving_average(predictions[:, 1], window_size)
smoothed_predictions_z = moving_average(predictions[:, 2], window_size)

# Adjust ground truth to match the smoothed prediction length
smoothed_ground_truth_x = ground_truth[window_size-1:, 0]
smoothed_ground_truth_y = ground_truth[window_size-1:, 1]
smoothed_ground_truth_z = ground_truth[window_size-1:, 2]

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


# Plot the smoothed 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ground_truth = ground_truth.numpy()
ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.plot(smoothed_predictions_x, smoothed_predictions_y, smoothed_predictions_z, label='Predictions (Smoothed)', color='blue')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Smoothed 3D Trajectory: Ground Truth vs Predictions')
ax.legend()
plt.show()
