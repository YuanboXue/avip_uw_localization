import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

# Load the IMU sequence data (10 time steps per DVL timestamp)
imu_sequence_data = pd.read_csv('../data/resam_imu_10step.csv')

# Load the single-time-step DVL, Pressure, and Ground Truth data
single_timestep_data = pd.read_csv('../data/synch_pool_shr_train01_iseq_apsingle.csv')

# Extract relevant columns
imu_sequences = imu_sequence_data[
    [
        'field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w',
        'field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z',
        'field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z'
    ]
].values.reshape(-1, 10, 10)  # Shape: (num_samples, sequence_length, IMU_feature_num)
dvl_data = single_timestep_data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
pressure_data = single_timestep_data[['pressure_fluid_pressure']].values
ground_truth = single_timestep_data[['gt_position_x', 'gt_position_y', 'gt_position_z',
                                     'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

# Split the data into training and testing sets
imu_train, imu_test, dvl_train, dvl_test, pressure_train, pressure_test, gt_train, gt_test = train_test_split(
    imu_sequences, dvl_data, pressure_data, ground_truth, test_size=0.2, random_state=42
)

# Normalize the data
# imu_sequences_mean = np.mean(imu_train, axis=(0, 1))
# imu_sequences_std = np.std(imu_train, axis=(0, 1))
# imu_train = (imu_train - imu_sequences_mean) / imu_sequences_std
# imu_test = (imu_test - imu_sequences_mean) / imu_sequences_std

# dvl_data_mean = np.mean(dvl_train, axis=0)
# dvl_data_std = np.std(dvl_train, axis=0)
# dvl_train = (dvl_train - dvl_data_mean) / dvl_data_std
# dvl_test = (dvl_test - dvl_data_mean) / dvl_data_std

# pressure_data_mean = np.mean(pressure_train, axis=0)
# pressure_data_std = np.std(pressure_train, axis=0)
# pressure_train = (pressure_train - pressure_data_mean) / pressure_data_std
# pressure_test = (pressure_test - pressure_data_mean) / pressure_data_std

# ground_truth_mean = np.mean(gt_train, axis=0)
# ground_truth_std = np.std(gt_train, axis=0)
# gt_train = (gt_train - ground_truth_mean) / ground_truth_std
# gt_test = (gt_test - ground_truth_mean) / ground_truth_std

# Convert data to PyTorch tensors
# imu_train = torch.tensor(imu_train, dtype=torch.float32)
# imu_test = torch.tensor(imu_test, dtype=torch.float32)
# dvl_train = torch.tensor(dvl_train, dtype=torch.float32)
# dvl_test = torch.tensor(dvl_test, dtype=torch.float32)
# pressure_train = torch.tensor(pressure_train, dtype=torch.float32)
# pressure_test = torch.tensor(pressure_test, dtype=torch.float32)
# gt_train = torch.tensor(gt_train, dtype=torch.float32)
# gt_test = torch.tensor(gt_test, dtype=torch.float32)

# Define the IMU Encoder (processes sequences)
class IMUEncoder(nn.Module):
    def __init__(self, input_dim=10, sequence_length=10, output_dim=64):
        super(IMUEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * sequence_length, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # Change shape to (batch_size, input_dim, sequence_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Define the DVL Encoder (processes single-time-step data)
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

# Define the Pressure Encoder (processes single-time-step data)
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
imu_sequences_mean = np.mean(imu_sequences, axis=(0, 1))
imu_sequences_std = np.std(imu_sequences, axis=(0, 1))
imu_sequences = (imu_sequences - imu_sequences_mean) / imu_sequences_std

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
imu_sequences = torch.tensor(imu_sequences, dtype=torch.float32)
dvl_data = torch.tensor(dvl_data, dtype=torch.float32)
pressure_data = torch.tensor(pressure_data, dtype=torch.float32)
ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

# Initialize the encoders and NKF
imu_encoder = IMUEncoder(input_dim=10, sequence_length=10, output_dim=64)
dvl_encoder = DVLEncoder(input_dim=3, output_dim=64)
pressure_encoder = PressureEncoder(input_dim=1, output_dim=64)
nkf = NeuralKalmanFilter(imu_encoder, dvl_encoder, pressure_encoder, hidden_dim=128, output_dim=7)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(nkf.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    outputs = nkf(imu_sequences, dvl_data, pressure_data)
    loss = criterion(outputs, ground_truth)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(nkf.state_dict(), 'nkf_imu_seq.pth')
print("Training complete. Model saved as 'nkf_imu_seq.pth'.")

# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()

#     # Forward pass
#     outputs = nkf(imu_train, dvl_train, pressure_train)
#     loss = criterion(outputs, gt_train)

#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
# torch.save(nkf.state_dict(), 'nkf_imu_seq_80%.pth')
# print("Training complete. Model saved as 'nkf_imu_seq_80%.pth'.")

# Evaluate the model
nkf.load_state_dict(torch.load('nkf_imu_seq.pth'))
nkf.eval()
with torch.no_grad():
    predictions = nkf(imu_sequences, dvl_data, pressure_data).numpy()
# with torch.no_grad():
#     predictions = nkf(imu_test, dvl_test, pressure_test).numpy()

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

# Plot the 3D trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ground_truth = ground_truth.numpy()
ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions', color='blue')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory: Ground Truth vs Predictions')
ax.legend()
plt.show()

# Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(gt_test[:, 0], gt_test[:, 1], label='Ground Truth', linestyle='--', color='orange')
# plt.plot(predictions[:, 0], predictions[:, 1], label='Predictions', color='blue')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('2D Trajectory: Ground Truth vs Predictions')
# plt.legend()
# plt.grid()
# plt.show()

# Plot the 3D trajectory
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# gt_test = gt_test.numpy()
# ax.plot(gt_test[:, 0], gt_test[:, 1], gt_test[:, 2], label='Ground Truth', linestyle='--', color='orange')
# ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions', color='blue')
# ax.set_xlabel('X Position (m)')
# ax.set_ylabel('Y Position (m)')
# ax.set_zlabel('Z Position (m)')
# ax.set_title('3D Trajectory: Ground Truth vs Predictions')
# ax.legend()
# plt.show()

# Traj Smoothing
# Option 1: Apply Savitzky-Golay filter to smooth the trajectory
window_length = 25  # Must be odd and greater than the polynomial order
poly_order = 3
smoothed_predictions_x = savgol_filter(predictions[:, 0], window_length, poly_order)
smoothed_predictions_y = savgol_filter(predictions[:, 1], window_length, poly_order)
smoothed_predictions_z = savgol_filter(predictions[:, 2], window_length, poly_order)

# Plot the smoothed trajectory
plt.figure(figsize=(10, 6))
plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', linestyle='--', color='orange')
plt.plot(smoothed_predictions_x, smoothed_predictions_y, label='Predictions (Smoothed)', color='blue')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Smoothed 2D Trajectory: Ground Truth vs Predictions')
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.plot(smoothed_predictions_x, smoothed_predictions_y, smoothed_predictions_z, label='Predictions (Smoothed)', color='blue')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Smoothed 3D Trajectory: Ground Truth vs Predictions')
ax.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(gt_test[:, 0], gt_test[:, 1], label='Ground Truth', linestyle='--', color='orange')
# plt.plot(smoothed_predictions_x, smoothed_predictions_y, label='Predictions (Smoothed)', color='blue')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title('Smoothed 2D Trajectory: Ground Truth vs Predictions')
# plt.legend()
# plt.grid()
# plt.show()

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(gt_test[:, 0], gt_test[:, 1], gt_test[:, 2], label='Ground Truth', linestyle='--', color='orange')
# ax.plot(smoothed_predictions_x, smoothed_predictions_y, smoothed_predictions_z, label='Predictions (Smoothed)', color='blue')
# ax.set_xlabel('X Position (m)')
# ax.set_ylabel('Y Position (m)')
# ax.set_zlabel('Z Position (m)')
# ax.set_title('Smoothed 3D Trajectory: Ground Truth vs Predictions')
# ax.legend()
# plt.show()
