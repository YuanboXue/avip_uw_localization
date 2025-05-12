import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the synchronized training and evaluation data
train_data = pd.read_csv('../../data/synch_pool_mix_check.csv')
eval_data = pd.read_csv('../../data/synch_pool_shr_train01.csv')

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
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model and predict changes
model.eval()
with torch.no_grad():
    predicted_changes = model(X_eval).numpy()

# Compute predicted trajectory
start_position = eval_data[position_cols].iloc[0].values
predicted_positions = np.cumsum(np.vstack((start_position, predicted_changes[:, :3])), axis=0)

# Plot ground truth and predicted trajectories
plt.figure(figsize=(10, 6))
plt.plot(eval_data['gt_position_x'], eval_data['gt_position_y'], label='Ground Truth', color='blue')
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted', color='red', linestyle='--')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Ground Truth vs Predicted Trajectory')
plt.legend()
plt.grid()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'transformer_rawinput_mix_1000.pth')
print("Training complete. Model saved as 'transformer_rawinput_mix_10000.pth'.")