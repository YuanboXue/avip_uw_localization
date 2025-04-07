import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# Load the synchronized CSV file
# 1) for each DVL timestamp, extracted a sequence of IMU data (10 time steps), if missing then interpolate; 
# 2) for each DVL timestamp, find the nearest pressure sensor data; 
# 3) for each DVL timestamp, find the nearest ground truth as the train label
imu_sequence_data = pd.read_csv('../data/resam_imu_10step.csv')
single_timestep_data = pd.read_csv('../data/synch_pool_shr_train01_iseq_apsingle.csv')

# Extract the relevant columns
imu_sequences = imu_sequence_data[
    [
        'field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w',
        'field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z',
        'field.linear_acceleration.x', 'field.linear_acceleration.y', 'field.linear_acceleration.z'
    ]
].values.reshape(-1, 10, 10)  # Shape: (num_samples, sequence_length, IMU_feature_num)
imu_orientations = imu_sequences[:, :, :4]  # Extract orientation (quaternion)
dvl_data = single_timestep_data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
pressure_data = single_timestep_data[['pressure_fluid_pressure']].values
ground_truth = single_timestep_data[['gt_position_x', 'gt_position_y', 'gt_position_z',
                                     'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

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
input_dim = imu_sequences.shape[2] + dvl_data.shape[1] + pressure_data.shape[1]
hidden_dim = 64
output_dim = ground_truth.shape[1]

# Initialize the model, loss function, and optimizer
model = NeuralKalmanFilter(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    imu_features = imu_sequences.view(imu_sequences.size(0), -1)  # Flatten IMU sequences
    inputs = torch.cat((imu_features, dvl_data, pressure_data), dim=1)
    outputs = model(inputs)
    loss = criterion(outputs, ground_truth)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'nkf_odoinput_seq.pth')
print("Training complete. Model saved as 'nkf_odoinput_seq.pth'.")

# Evaluate the model
model.eval()
with torch.no_grad():
    imu_features = imu_sequences.view(imu_sequences.size(0), -1)  # Flatten IMU sequences
    inputs = torch.cat((imu_features, dvl_data, pressure_data), dim=1)
    predictions = model(inputs).numpy()

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
ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linestyle='--', color='orange')
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions', color='blue')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory: Ground Truth vs Predictions')
ax.legend()
plt.show()
