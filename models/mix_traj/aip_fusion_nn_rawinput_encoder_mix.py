import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the training data
train_data = pd.read_csv('../data/synch_pool_mix.csv')

# Define the encoders for each sensor
class IMUEncoder(nn.Module):
    def __init__(self):
        super(IMUEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.fc(x)

class DVLEncoder(nn.Module):
    def __init__(self):
        super(DVLEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.fc(x)

class PressureEncoder(nn.Module):
    def __init__(self):
        super(PressureEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.fc(x)

# Define the neural network for vehicle localization
class VehicleLocalizationNN(nn.Module):
    def __init__(self):
        super(VehicleLocalizationNN, self).__init__()
        self.imu_encoder = IMUEncoder()
        self.dvl_encoder = DVLEncoder()
        self.pressure_encoder = PressureEncoder()
        self.fc = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Output: position (3) + orientation (4)
        )

    def forward(self, imu_data, dvl_data, pressure_data):
        imu_features = self.imu_encoder(imu_data)
        dvl_features = self.dvl_encoder(dvl_data)
        pressure_features = self.pressure_encoder(pressure_data)
        concatenated_features = torch.cat((imu_features, dvl_features, pressure_features), dim=1)
        output = self.fc(concatenated_features)
        return output

# Prepare the training data
imu_data = train_data.iloc[:, 4:14].values
dvl_data = train_data.iloc[:, 1:4].values
pressure_data = train_data.iloc[:, 14:15].values
ground_truth = train_data.iloc[:, 15:22].values

# Normalize the input data
scaler_imu = StandardScaler()
scaler_dvl = StandardScaler()
scaler_pressure = StandardScaler()

imu_data = scaler_imu.fit_transform(imu_data)
dvl_data = scaler_dvl.fit_transform(dvl_data)
pressure_data = scaler_pressure.fit_transform(pressure_data)

imu_data = torch.tensor(imu_data, dtype=torch.float32)
dvl_data = torch.tensor(dvl_data, dtype=torch.float32)
pressure_data = torch.tensor(pressure_data, dtype=torch.float32)
ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

# Check for NaN values in the input data and replace them with zeros
imu_data[torch.isnan(imu_data)] = 0
dvl_data[torch.isnan(dvl_data)] = 0
pressure_data[torch.isnan(pressure_data)] = 0
ground_truth[torch.isnan(ground_truth)] = 0

# Initialize the model, loss function and optimizer
model = VehicleLocalizationNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(imu_data, dvl_data, pressure_data)
    loss = criterion(outputs, ground_truth)
    
    # Check for NaN values in the loss and break if found
    if torch.isnan(loss).any():
        print(f'NaN loss encountered at epoch {epoch+1}')
        break
    
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Load the evaluation data
eval_data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Prepare the evaluation data
imu_eval_data = eval_data.iloc[:, 4:14].values
dvl_eval_data = eval_data.iloc[:, 1:4].values
pressure_eval_data = eval_data.iloc[:, 14:15].values
gt_position_eval = eval_data.iloc[:, 15:18].values

# Normalize the evaluation data using the same scalers as training data
imu_eval_data = scaler_imu.transform(imu_eval_data)
dvl_eval_data = scaler_dvl.transform(dvl_eval_data)
pressure_eval_data = scaler_pressure.transform(pressure_eval_data)

imu_eval_data = torch.tensor(imu_eval_data, dtype=torch.float32)
dvl_eval_data = torch.tensor(dvl_eval_data, dtype=torch.float32)
pressure_eval_data = torch.tensor(pressure_eval_data, dtype=torch.float32)

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(imu_eval_data, dvl_eval_data, pressure_eval_data).numpy()

# Extract ground truth and predicted positions
gt_positions = gt_position_eval
pred_positions = predictions[:, :3]

# Plot the ground truth and predicted trajectories
plt.figure(figsize=(10, 6))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth')
plt.plot(pred_positions[:, 0], pred_positions[:, 1], label='Predicted')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Ground Truth vs Predicted Trajectory')
plt.legend()
plt.show()
