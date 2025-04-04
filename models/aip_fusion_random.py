import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd

class InertialEncoder(nn.Module):
    def __init__(self):
        super(InertialEncoder, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 10, 64)
        self.fc_cov = nn.Linear(64 * 10, 64)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape from (B, 10, 6) to (B, 6, 10)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        feature = self.fc(x)
        noise_cov = self.fc_cov(x)
        return feature, noise_cov
    
class DVLEncoder(nn.Module):
    def __init__(self):
        super(DVLEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_cov = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        feature = self.relu(self.bn2(self.fc2(x)))
        noise_cov = self.fc_cov(x)
        return feature, noise_cov
    
class PressureEncoder(nn.Module):
    def __init__(self):
        super(PressureEncoder, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_cov = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        feature = self.relu(self.bn2(self.fc2(x)))
        noise_cov = self.fc_cov(x)
        return feature, noise_cov
    
class SensorFusion(nn.Module):
    def __init__(self):
        super(SensorFusion, self).__init__()
        self.fc = nn.Linear(192, 192)  # Adjust input size to match the concatenated features

    def forward(self, inertial_feature, dvl_feature, pressure_feature):
        x = torch.cat((inertial_feature, dvl_feature, pressure_feature), dim=1)
        observation = self.fc(x)
        return observation
    
class NeuralTransitionGeneration(nn.Module):
    def __init__(self):
        super(NeuralTransitionGeneration, self).__init__()
        self.fc1 = nn.Linear(192, 512)
        self.fc2 = nn.Linear(512, 192)
        self.fc_A = nn.Linear(192, 192 * 192)
        self.fc_cov = nn.Linear(192, 192 * 192)
        self.relu = nn.ReLU()

    def forward(self, latent_state):
        x = self.relu(self.fc1(latent_state))
        x = self.relu(self.fc2(x))
        transition_matrix = self.fc_A(x).view(-1, 192, 192)
        process_noise_cov = self.fc_cov(x).view(-1, 192, 192)
        return transition_matrix, process_noise_cov
    
class KalmanFilter(nn.Module):
    def __init__(self):
        super(KalmanFilter, self).__init__()

    def forward(self, prev_latent_states, prev_state_cov, transition_A, process_noise, observation, observation_noise):
        # Prediction step
        predicted_state = torch.matmul(transition_A, prev_latent_states.unsqueeze(-1)).squeeze(-1)
        predicted_cov = torch.matmul(torch.matmul(transition_A, prev_state_cov), transition_A.transpose(-1, -2)) + process_noise

        # Update step
        innovation = observation - predicted_state
        innovation_cov = predicted_cov + observation_noise
        kalman_gain = torch.matmul(predicted_cov, torch.inverse(innovation_cov))
        current_latent_states = predicted_state + torch.matmul(kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)
        current_state_cov = predicted_cov - torch.matmul(torch.matmul(kalman_gain, innovation_cov), kalman_gain.transpose(-1, -2))

        return current_latent_states, current_state_cov
    
class PosePredictor(nn.Module):
    def __init__(self):
        super(PosePredictor, self).__init__()
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 6)
        self.relu = nn.ReLU()

    def forward(self, latent_states):
        x = self.relu(self.fc1(latent_states))
        poses = self.fc2(x)
        return poses
    
class UnderwaterLocalizationSystem(nn.Module):
    def __init__(self):
        super(UnderwaterLocalizationSystem, self).__init__()
        self.inertial_encoder = InertialEncoder()
        self.dvl_encoder = DVLEncoder()
        self.pressure_encoder = PressureEncoder()
        self.sensor_fusion = SensorFusion()
        self.transition_generation = NeuralTransitionGeneration()
        self.kalman_filter = KalmanFilter()
        self.pose_predictor = PosePredictor()

    def forward(self, imu_data, dvl_data, pressure_data, prev_latent_states, prev_state_cov):
        # Encode sensor data
        inertial_feature, inertial_noise_cov = self.inertial_encoder(imu_data)
        dvl_feature, dvl_noise_cov = self.dvl_encoder(dvl_data)
        pressure_feature, pressure_noise_cov = self.pressure_encoder(pressure_data)

        # Fuse sensor features
        observation = self.sensor_fusion(inertial_feature, dvl_feature, pressure_feature)

        # Generate transition matrix and process noise
        transition_A, process_noise = self.transition_generation(prev_latent_states)

        # Apply Kalman filter
        current_latent_states, current_state_cov = self.kalman_filter(
            prev_latent_states, prev_state_cov, transition_A, process_noise, observation, observation_noise
        )

        # Predict poses
        poses = self.pose_predictor(current_latent_states)

        return poses, current_latent_states, current_state_cov

class RandomSensorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imu = torch.randn(10, 6)  # Random IMU data
        dvl = torch.randn(3)  # Random DVL data
        pressure = torch.randn(1)  # Random pressure data
        pose = torch.randn(6)  # Random ground truth pose
        return imu, dvl, pressure, pose

# Create dataset with random data
num_samples = 1000
dataset = RandomSensorDataset(num_samples)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = UnderwaterLocalizationSystem()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (imu_data, dvl_data, pressure_data, ground_truth_poses) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        prev_latent_states = torch.randn(batch_size, 192)
        prev_state_cov = torch.eye(192).unsqueeze(0).repeat(batch_size, 1, 1)
        observation_noise = torch.eye(192).unsqueeze(0).repeat(batch_size, 1, 1)
        poses, _, _ = model(imu_data, dvl_data, pressure_data, prev_latent_states, prev_state_cov)

        # Compute loss
        loss = criterion(poses, ground_truth_poses)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imu_data, dvl_data, pressure_data, ground_truth_poses in val_loader:
            prev_latent_states = torch.randn(batch_size, 192)
            prev_state_cov = torch.eye(192).unsqueeze(0).repeat(batch_size, 1, 1)
            observation_noise = torch.eye(192).unsqueeze(0).repeat(batch_size, 1, 1)
            poses, _, _ = model(imu_data, dvl_data, pressure_data, prev_latent_states, prev_state_cov)
            loss = criterion(poses, ground_truth_poses)
            val_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}')

print('Training complete')
