import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import datetime

# Load the synchronized training and evaluation data
train_data = pd.read_csv('../../data/synch_pool_mix_check.csv')
# eval_data = pd.read_csv('../../data/synch_pool_shr_train01.csv')
# eval_data = pd.read_csv('../../data/synch_pool_zz_train02.csv')
# eval_data = pd.read_csv('../../data/synch_pool_zz_train01.csv')
eval_data = pd.read_csv('../../data/synch_ship_cir_train01.csv')

# Define features and target columns
# features = ['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z',
#             'imu_orientation_w', 'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z',
#             'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
#             'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z',
#             'pressure_fluid_pressure']
features = ['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z',
            'imu_orientation_w', 'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z',
            'pressure_fluid_pressure']
position_cols = ['gt_position_x', 'gt_position_y', 'gt_position_z']
orientation_cols = ['gt_orientation_w', 'gt_orientation_x', 'gt_orientation_y', 'gt_orientation_z']

# Preprocess the data to compute changes
def compute_changes(data, position_cols, orientation_cols):
    position_changes = data[position_cols].diff().fillna(0).values
    orientation_changes = data[orientation_cols].diff().fillna(0).values
    return np.hstack((position_changes, orientation_changes))

train_changes = compute_changes(train_data, position_cols, orientation_cols)
eval_changes = compute_changes(eval_data, position_cols, orientation_cols)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
X_eval = scaler.transform(eval_data[features])

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train = torch.tensor(train_changes, dtype=torch.float32)
X_eval = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1)
y_eval = torch.tensor(eval_changes, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout_rate):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc1(x[:, -1, :])
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[2]
hidden_size = 256
output_size = y_train.shape[1]
num_layers = 2
num_heads = 4
dropout_rate = 0.1

# Ensure input_size is divisible by num_heads
# num_heads = 4
# if input_size % num_heads != 0:
#     raise ValueError(f"input_size ({input_size}) must be divisible by num_heads ({num_heads}).")

model = TransformerModel(input_size, hidden_size, output_size, num_layers, num_heads, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
# num_epochs = 10000
# for epoch in range(num_epochs):
#     model.train()
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model and predict changes
model.load_state_dict(torch.load('transformer_rawinput_mix_1000.pth'))
model.eval()
with torch.no_grad():
    predicted_changes = model(X_eval).numpy()

# Compute predicted trajectory
start_position = eval_data[position_cols].iloc[0].values
start_orientation = eval_data[orientation_cols].iloc[0].values

# Get ground truth trajectories
gt_positions = eval_data[position_cols].values
gt_orientations = eval_data[orientation_cols].values

# Initialize arrays for predicted positions and orientations
predicted_positions = np.zeros((len(predicted_changes) + 1, 3))
predicted_orientations = np.zeros((len(predicted_changes) + 1, 4))

# Set initial values
predicted_positions[0] = start_position
predicted_orientations[0] = start_orientation

# Function to normalize quaternion
def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return q

# Build predicted trajectory from changes
for i in range(len(predicted_changes)):
    # Update positions with predicted position changes
    predicted_positions[i+1] = predicted_positions[i] + predicted_changes[i, :3]
    
    # Update orientations with predicted orientation changes and normalize
    predicted_orientations[i+1] = predicted_orientations[i] + predicted_changes[i, 3:7]
    predicted_orientations[i+1] = normalize_quaternion(predicted_orientations[i+1])

# Convert quaternions to Euler angles for visualization
def quaternion_to_euler_angles(quaternions):
    """Convert array of quaternions to Euler angles (roll, pitch, yaw)."""
    euler_angles = np.zeros((quaternions.shape[0], 3))
    
    for i, quat in enumerate(quaternions):
        # Convert to scipy Rotation (note: scipy uses xyzw order)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Extract [x,y,z,w]
        # Get Euler angles in degrees
        euler_angles[i] = r.as_euler('xyz', degrees=True)
        
    return euler_angles

# Get Euler angles
gt_euler_angles = quaternion_to_euler_angles(gt_orientations)
pred_euler_angles = quaternion_to_euler_angles(predicted_orientations[1:])  # Skip initial orientation

# Define labels for Euler angles
euler_labels = ['Roll', 'Pitch', 'Yaw']

# Plot position comparison
plt.figure(figsize=(12, 8))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', color='blue')
plt.plot(predicted_positions[1:, 0], predicted_positions[1:, 1], label='Predicted', color='red', linestyle='--')
plt.scatter(gt_positions[0, 0], gt_positions[0, 1], color='blue', s=100, marker='o', label='Start')
plt.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color='blue', s=100, marker='x', label='End')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Ground Truth vs Predicted Trajectory')
plt.legend()
plt.grid(True)
# plt.savefig('position_trajectory_zz.png')
plt.show()

# 3D trajectory plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='blue')
ax.plot(predicted_positions[1:, 0], predicted_positions[1:, 1], predicted_positions[1:, 2], 
        label='Predicted', color='red', linestyle='--')
ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], color='blue', s=100, marker='o', label='Start')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory Comparison')
ax.legend()
# plt.savefig('position_trajectory_3d_zz.png')
plt.show()

# Plot orientation comparison (Euler angles)
plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(gt_euler_angles[:, i], label='Ground Truth', color='blue')
    plt.plot(pred_euler_angles[:, i], label='Predicted', color='red', linestyle='--')
    plt.ylabel(f'{euler_labels[i]} (degrees)')
    plt.title(f'{euler_labels[i]} Comparison')
    plt.grid(True)
    if i == 0:
        plt.legend()
    if i == 2:
        plt.xlabel('Time Step')
plt.tight_layout()
# plt.savefig('orientation_euler_comparison_zz.png')
plt.show()

# Calculate position errors
position_error = np.sqrt(np.sum((predicted_positions[1:] - gt_positions) ** 2, axis=1))

# Calculate orientation errors using quaternion distance
def quaternion_angular_distance(q1, q2):
    """Calculate angular distance between quaternions in degrees."""
    # Convert to scipy Rotation objects
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])  # [x,y,z,w]
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])  # [x,y,z,w]
    
    # Calculate relative rotation and extract angle
    r_diff = r1.inv() * r2
    angle = np.abs(r_diff.magnitude() * (180.0 / np.pi))  # Convert to degrees
    return angle

orientation_error = np.array([
    quaternion_angular_distance(predicted_orientations[i+1], gt_orientations[i])
    for i in range(len(gt_orientations))
])

# Plot errors
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(position_error)
plt.ylabel('Position Error (m)')
plt.title('Position Error Over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(orientation_error)
plt.xlabel('Time Step')
plt.ylabel('Orientation Error (degrees)')
plt.title('Orientation Error Over Time')
plt.grid(True)
plt.tight_layout()
# plt.savefig('error_metrics_zz.png')
plt.show()

# Print error statistics
print("\nPosition Error Statistics:")
print(f"Mean: {np.mean(position_error):.4f} m")
print(f"RMSE: {np.sqrt(np.mean(position_error**2)):.4f} m")
print(f"Max: {np.max(position_error):.4f} m")

print("\nOrientation Error Statistics:")
print(f"Mean: {np.mean(orientation_error):.4f} degrees")
print(f"RMSE: {np.sqrt(np.mean(orientation_error**2)):.4f} degrees")
print(f"Max: {np.max(orientation_error):.4f} degrees")

# Save the trained model
# torch.save(model.state_dict(), 'transformer_rawinput_mix_1000.pth')
# print("Training complete. Model saved as 'transformer_rawinput_mix_10000.pth'.")

# Save results in TUM format: timestamp tx ty tz qx qy qz qw
# Get timestamps
timestamps = eval_data['timestamp'].values
if isinstance(timestamps[0], str):
    # Try to convert string timestamps to numeric
    try:
        timestamps = pd.to_datetime(timestamps).astype(np.int64) // 10**9  # Convert to seconds
    except:
        # If conversion fails, use sequential timestamps
        timestamps = np.arange(len(gt_positions))

# Create TUM format dataframes (ground truth)
tum_groundtruth = pd.DataFrame({
    'timestamp': timestamps,
    'tx': gt_positions[:, 0],
    'ty': gt_positions[:, 1],
    'tz': gt_positions[:, 2],
    'qx': gt_orientations[:, 1],  # TUM format: qx qy qz qw
    'qy': gt_orientations[:, 2],
    'qz': gt_orientations[:, 3],
    'qw': gt_orientations[:, 0]
})

# Create TUM format dataframe (predictions)
tum_predicted = pd.DataFrame({
    'timestamp': timestamps,
    'tx': predicted_positions[1:, 0],
    'ty': predicted_positions[1:, 1],
    'tz': predicted_positions[1:, 2],
    'qx': predicted_orientations[1:, 1],
    'qy': predicted_orientations[1:, 2],
    'qz': predicted_orientations[1:, 3],
    'qw': predicted_orientations[1:, 0]
})

# Save TUM format files
# date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# tum_groundtruth.to_csv(f'groundtruth_tum_zz_{date_str}.txt', sep=' ', index=False, header=False)
# tum_predicted.to_csv(f'predicted_tum_zz_{date_str}.txt', sep=' ', index=False, header=False)
# print(f"Saved trajectory data in TUM format with timestamp {date_str}")