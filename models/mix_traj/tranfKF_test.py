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
from filterpy.kalman import KalmanFilter

# Load the synchronized training and evaluation data
train_data = pd.read_csv('../../data/synch_pool_mix_check.csv')
eval_data = pd.read_csv('../../data/synch_pool_shr_train01.csv')

print(f"Loaded training data: {train_data.shape}, evaluation data: {eval_data.shape}")

# Define features and target columns
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

model = TransformerModel(input_size, hidden_size, output_size, num_layers, num_heads, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to initialize position Kalman filter
def init_position_kf(initial_pos, dt=0.1):
    """
    Initialize Kalman filter for position tracking.
    State vector: [x, y, z, vx, vy, vz]
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)
    
    # State transition matrix F
    kf.F = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    # Measurement matrix H
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    
    # Measurement noise covariance R
    kf.R = np.eye(3) * 0.1
    
    # Process noise covariance Q
    q = 0.01
    kf.Q = np.eye(6) * q
    kf.Q[3:, 3:] *= 10  # Higher noise for velocity components
    
    # Initial state
    kf.x = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0, 0, 0]).reshape(6, 1)
    
    # Initial state covariance
    kf.P = np.eye(6) * 1.0
    
    return kf

# Function to initialize orientation Kalman filter
def init_orientation_kf(initial_quat):
    """
    Initialize Kalman filter for orientation tracking using quaternions.
    State vector: [qw, qx, qy, qz]
    """
    kf = KalmanFilter(dim_x=4, dim_z=4)
    
    # State transition matrix (identity for quaternion)
    kf.F = np.eye(4)
    
    # Measurement matrix
    kf.H = np.eye(4)
    
    # Measurement noise covariance R
    kf.R = np.eye(4) * 0.01
    
    # Process noise covariance Q
    kf.Q = np.eye(4) * 0.001
    
    # Initial state
    kf.x = initial_quat.reshape(4, 1)
    
    # Initial state covariance
    kf.P = np.eye(4) * 0.1
    
    return kf

# Function to normalize quaternion
def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return q

# Function to update predicted trajectory with Kalman filter
def apply_kalman_filter(positions, orientations, dt=0.1):
    """
    Apply Kalman filtering to smooth position and orientation trajectories.
    """
    # Initialize Kalman filters
    position_kf = init_position_kf(positions[0], dt)
    orientation_kf = init_orientation_kf(orientations[0])
    
    filtered_positions = np.zeros_like(positions)
    filtered_orientations = np.zeros_like(orientations)
    
    # Set initial state
    filtered_positions[0] = positions[0]
    filtered_orientations[0] = orientations[0]
    
    # Process each timestep
    for i in range(1, len(positions)):
        # Predict
        position_kf.predict()
        orientation_kf.predict()
        
        # Update with measurement
        position_kf.update(positions[i])
        
        # Normalize quaternion before update
        norm_quat = normalize_quaternion(orientations[i])
        orientation_kf.update(norm_quat)
        
        # Store filtered values
        filtered_positions[i] = position_kf.x[:3, 0]
        
        # Normalize quaternion after filtering
        filtered_quat = orientation_kf.x[:, 0]
        filtered_orientations[i] = normalize_quaternion(filtered_quat)
    
    return filtered_positions, filtered_orientations

# Train the model
print("Starting training...")
num_epochs = 100
train_losses = []
eval_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    avg_loss = epoch_loss / batch_count
    train_losses.append(avg_loss)
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        eval_output = model(X_eval)
        eval_loss = criterion(eval_output, y_eval).item()
        eval_losses.append(eval_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}')

# Plot training and evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
plt.show()

# Evaluate the model and predict changes
model.eval()
with torch.no_grad():
    predicted_changes = model(X_eval).numpy()

# Compute predicted trajectory using cumulative sum of changes
start_position = eval_data[position_cols].iloc[0].values
start_orientation = eval_data[orientation_cols].iloc[0].values

# Get ground truth trajectories
gt_positions = eval_data[position_cols].values
gt_orientations = eval_data[orientation_cols].values

# Build transformer (unfiltered) trajectories
transformer_positions = np.zeros((len(predicted_changes) + 1, 3))
transformer_orientations = np.zeros((len(predicted_changes) + 1, 4))

transformer_positions[0] = start_position
transformer_orientations[0] = start_orientation

for i in range(len(predicted_changes)):
    # Update positions with predicted changes
    transformer_positions[i+1] = transformer_positions[i] + predicted_changes[i, :3]
    
    # Update orientations with predicted changes and normalize
    transformer_orientations[i+1] = transformer_orientations[i] + predicted_changes[i, 3:7]
    transformer_orientations[i+1] = normalize_quaternion(transformer_orientations[i+1])

# Apply Kalman filtering to the transformer predictions
print("Applying Kalman filter to transformer predictions...")
filtered_positions, filtered_orientations = apply_kalman_filter(
    transformer_positions, transformer_orientations, dt=0.1
)

# Calculate trajectory errors
def calculate_position_error(pred, gt):
    """Calculate Euclidean distance error between predicted and ground truth positions"""
    return np.sqrt(np.sum((pred - gt) ** 2, axis=1))

def calculate_orientation_error(pred_quat, gt_quat):
    """Calculate angular distance between two quaternions in degrees"""
    errors = []
    for i in range(len(pred_quat)):
        # Convert to scipy Rotation objects (note the order difference)
        r_pred = R.from_quat([pred_quat[i, 1], pred_quat[i, 2], pred_quat[i, 3], pred_quat[i, 0]])  # [x,y,z,w]
        r_gt = R.from_quat([gt_quat[i, 1], gt_quat[i, 2], gt_quat[i, 3], gt_quat[i, 0]])
        
        # Calculate relative rotation and extract angle
        r_diff = r_pred.inv() * r_gt
        angle = np.abs(r_diff.magnitude() * (180.0 / np.pi))  # Convert to degrees
        errors.append(angle)
    return np.array(errors)

# Calculate errors
transformer_pos_error = calculate_position_error(transformer_positions[1:], gt_positions)
filtered_pos_error = calculate_position_error(filtered_positions[1:], gt_positions)

transformer_orient_error = calculate_orientation_error(transformer_orientations[1:], gt_orientations)
filtered_orient_error = calculate_orientation_error(filtered_orientations[1:], gt_orientations)

# Print error statistics
print("\nPosition Error Statistics (meters):")
print(f"Transformer - Mean: {np.mean(transformer_pos_error):.4f}, RMSE: {np.sqrt(np.mean(transformer_pos_error**2)):.4f}")
print(f"Kalman Filter - Mean: {np.mean(filtered_pos_error):.4f}, RMSE: {np.sqrt(np.mean(filtered_pos_error**2)):.4f}")
print(f"Improvement: {(1 - np.mean(filtered_pos_error)/np.mean(transformer_pos_error))*100:.2f}%")

print("\nOrientation Error Statistics (degrees):")
print(f"Transformer - Mean: {np.mean(transformer_orient_error):.4f}, RMSE: {np.sqrt(np.mean(transformer_orient_error**2)):.4f}")
print(f"Kalman Filter - Mean: {np.mean(filtered_orient_error):.4f}, RMSE: {np.sqrt(np.mean(filtered_orient_error**2)):.4f}")
print(f"Improvement: {(1 - np.mean(filtered_orient_error)/np.mean(transformer_orient_error))*100:.2f}%")

# Plot ground truth and predicted trajectories
plt.figure(figsize=(12, 8))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', color='blue')
plt.plot(transformer_positions[1:, 0], transformer_positions[1:, 1], label='Transformer', color='red', linestyle='--')
plt.plot(filtered_positions[1:, 0], filtered_positions[1:, 1], label='Kalman Filtered', color='green')
plt.scatter(gt_positions[0, 0], gt_positions[0, 1], color='blue', s=100, marker='o', label='Start')
plt.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color='blue', s=100, marker='x', label='End')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Trajectory Comparison')
plt.legend()
plt.grid(True)
plt.savefig('trajectory_2d_comparison.png')
plt.show()

# 3D trajectory plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='blue')
ax.plot(transformer_positions[1:, 0], transformer_positions[1:, 1], transformer_positions[1:, 2], 
        label='Transformer', color='red', linestyle='--')
ax.plot(filtered_positions[1:, 0], filtered_positions[1:, 1], filtered_positions[1:, 2], 
        label='Kalman Filtered', color='green')
ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], color='blue', s=100, marker='o', label='Start')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Trajectory Comparison')
ax.legend()
plt.savefig('trajectory_3d_comparison.png')
plt.show()

# Plot position errors
plt.figure(figsize=(12, 6))
plt.plot(transformer_pos_error, label='Transformer', alpha=0.7)
plt.plot(filtered_pos_error, label='Kalman Filtered', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Position Error (m)')
plt.title('Position Error Comparison')
plt.legend()
plt.grid(True)
plt.savefig('position_error_comparison.png')
plt.show()

# Plot orientation errors
plt.figure(figsize=(12, 6))
plt.plot(transformer_orient_error, label='Transformer', alpha=0.7)
plt.plot(filtered_orient_error, label='Kalman Filtered', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Orientation Error (degrees)')
plt.title('Orientation Error Comparison')
plt.legend()
plt.grid(True)
plt.savefig('orientation_error_comparison.png')
plt.show()

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

# Create TUM format dataframes
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

tum_transformer = pd.DataFrame({
    'timestamp': timestamps,
    'tx': transformer_positions[1:, 0],
    'ty': transformer_positions[1:, 1],
    'tz': transformer_positions[1:, 2],
    'qx': transformer_orientations[1:, 1],
    'qy': transformer_orientations[1:, 2],
    'qz': transformer_orientations[1:, 3],
    'qw': transformer_orientations[1:, 0]
})

tum_filtered = pd.DataFrame({
    'timestamp': timestamps,
    'tx': filtered_positions[1:, 0],
    'ty': filtered_positions[1:, 1],
    'tz': filtered_positions[1:, 2],
    'qx': filtered_orientations[1:, 1],
    'qy': filtered_orientations[1:, 2],
    'qz': filtered_orientations[1:, 3],
    'qw': filtered_orientations[1:, 0]
})

# Save TUM format files
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tum_groundtruth.to_csv(f'groundtruth_tum_{date_str}.txt', sep=' ', index=False, header=False)
tum_transformer.to_csv(f'transformer_tum_{date_str}.txt', sep=' ', index=False, header=False)
tum_filtered.to_csv(f'filtered_tum_{date_str}.txt', sep=' ', index=False, header=False)
print(f"Saved trajectory data in TUM format with timestamp {date_str}")

# Save the combined data for analysis
combined_data = pd.DataFrame({
    'timestamp': timestamps,
    'gt_x': gt_positions[:, 0], 'gt_y': gt_positions[:, 1], 'gt_z': gt_positions[:, 2],
    'trans_x': transformer_positions[1:, 0], 'trans_y': transformer_positions[1:, 1], 'trans_z': transformer_positions[1:, 2],
    'kf_x': filtered_positions[1:, 0], 'kf_y': filtered_positions[1:, 1], 'kf_z': filtered_positions[1:, 2],
    'pos_err_trans': transformer_pos_error, 'pos_err_kf': filtered_pos_error,
    'orient_err_trans': transformer_orient_error, 'orient_err_kf': filtered_orient_error
})
combined_data.to_csv(f'trajectory_analysis_{date_str}.csv', index=False)
print(f"Saved combined analysis data to trajectory_analysis_{date_str}.csv")

# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'eval_loss': eval_losses[-1],
    'scaler': scaler,
}, f'transformer_rawinput_mix_kf_{date_str}.pth')

print(f"Training complete. Model saved as transformer_rawinput_mix_kf_{date_str}.pth.")