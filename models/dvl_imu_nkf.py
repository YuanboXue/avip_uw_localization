import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

# Load the data
imu_data = pd.read_csv('imu_davepool.csv')
dvl_data = pd.read_csv('dvl_davepool.csv')
gt_data = pd.read_csv('gt_davepool.csv')

# Convert timestamps to a common format if necessary
imu_data['timestamp'] = pd.to_datetime(imu_data['%time'])
dvl_data['timestamp'] = pd.to_datetime(dvl_data['timestamp'])
gt_data['timestamp'] = pd.to_datetime(gt_data['timestamp'])

# Interpolate DVL data to match IMU timestamps
dvl_interp = interp1d(dvl_data['timestamp'].astype(np.int64), dvl_data[['vx', 'vy', 'vz']], axis=0, fill_value="extrapolate")
dvl_resampled = dvl_interp(imu_data['timestamp'].astype(np.int64))

# Interpolate ground truth data to match IMU timestamps
gt_interp = interp1d(gt_data['timestamp'].astype(np.int64), gt_data[['x', 'y', 'z', 'vx', 'vy', 'vz']], axis=0, fill_value="extrapolate")
gt_resampled = gt_interp(imu_data['timestamp'].astype(np.int64))

# Define a simple dataset class
class SensorDataset(Dataset):
    def __init__(self, imu_data, dvl_data, gt_data):
        self.imu_data = imu_data
        self.dvl_data = dvl_data
        self.gt_data = gt_data

    def __len__(self):
        return len(self.imu_data)

    def __getitem__(self, idx):
        imu_sample = self.imu_data.iloc[idx]
        dvl_sample = self.dvl_data[idx]
        gt_sample = self.gt_data[idx]
        return (imu_sample[['field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w']].values.astype(np.float32),
                dvl_sample.astype(np.float32),
                gt_sample.astype(np.float32))
 
# Create dataset and dataloader
dataset = SensorDataset(imu_data, dvl_resampled, gt_resampled)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the neural network models
class IMUEncoder(nn.Module):
    def __init__(self):
        super(IMUEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, imu):
        return self.fc(imu)

class DVLEncoder(nn.Module):
    def __init__(self):
        super(DVLEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, dvl):
        return self.fc(dvl)

class TransitionModel(nn.Module):
    def __init__(self):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, imu_features, dvl_features):
        x = torch.cat((imu_features, dvl_features), dim=1)
        return self.fc(x)

# class ObservationModel(nn.Module):
#     def __init__(self):
#         super(ObservationModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(3, 64),  # 3 (DVL velocity)
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3)
#         )

#     def forward(self, dvl):
#         return self.fc(dvl)

# # Initialize models
# transition_model = TransitionModel()
# observation_model = ObservationModel()

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(list(transition_model.parameters()) + list(observation_model.parameters()), lr=0.001)

# # Training loop
# for epoch in range(10):  # Number of epochs
#     for imu, dvl, gt in dataloader:
#         # Initial state (could be improved with better initialization)
#         state = torch.zeros(imu.size(0), 10)

#         # Predict step
#         predicted_state = transition_model(state, imu)

#         # Update step
#         observed_state = observation_model(dvl)
#         state = predicted_state + observed_state

#         # Compute loss
#         loss = criterion(state, gt)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# print("Training complete.")

# Initialize models
imu_encoder = IMUEncoder()
dvl_encoder = DVLEncoder()
transition_model = TransitionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(imu_encoder.parameters()) + list(dvl_encoder.parameters()) + list(transition_model.parameters()), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    for imu, dvl, gt in dataloader:
        # Encode IMU and DVL data
        imu_features = imu_encoder(imu)
        dvl_features = dvl_encoder(dvl)

        # Predict step
        predicted_state = transition_model(imu_features, dvl_features)

        # Compute loss
        loss = criterion(predicted_state, gt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Training complete.")