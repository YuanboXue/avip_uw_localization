import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Load the synchronized CSV file
# 1) for each DVL timestamp, use interpolation to resample the IMU data; 
# 2) for each DVL timestamp, find the nearest pressure sensor data; 
# 3) for each DVL timestamp, find the nearest ground truth as the train label
data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Extract the relevant columns
imu_orientation = data[['imu_orientation_w', 'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z']].values
imu_linear_acc = data[['imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z']].values
imu_angular_vel = data[['imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z']].values
dvl_data = data[['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z']].values
pressure_data = data['pressure_fluid_pressure'].values
ground_truth = data[['gt_position_x', 'gt_position_y', 'gt_position_z', 'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z', 'gt_orientation_w']].values

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
input_dim = imu_orientation.shape[1] + imu_linear_acc.shape[1] + imu_angular_vel.shape[1] + dvl_data.shape[1] + 1  # IMU orientation + IMU linear acc + IMU angular vel + DVL + Pressure
hidden_dim = 64
output_dim = ground_truth.shape[1]  # Position and orientation quaternions


# Initialize the model, loss function, and optimizer
model = NeuralKalmanFilter(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the input and output data
inputs = np.hstack((imu_orientation, imu_linear_acc, imu_angular_vel, dvl_data, pressure_data.reshape(-1, 1)))
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(ground_truth, dtype=torch.float32)


# Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
    
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'neural_kalman_filter.pth')

# print("Training complete. Model saved as 'neural_kalman_filter.pth'.")

# Load the trained model
model = NeuralKalmanFilter(input_dim, hidden_dim, output_dim)
# model.load_state_dict(torch.load('neural_kalman_filter.pth'))
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
ax1.set_title('Position')
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
