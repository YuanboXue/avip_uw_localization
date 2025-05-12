import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the synchronized training and evaluation data
train_data = pd.read_csv('../data/synch_pool_mix_check.csv')
eval_data = pd.read_csv('../data/synch_pool_shr_train01.csv')

# Define features and target columns
features = ['dvl_velocity_x', 'dvl_velocity_y', 'dvl_velocity_z',
            'imu_orientation_w', 'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z',
            'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
            'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z',
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

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Grid search over hyperparameters
hidden_sizes = [64, 128, 256]
dropout_rates = [0.3, 0.5, 0.7]
learning_rates = [0.001, 0.0005]
best_loss = float('inf')
best_params = None

for hidden_size in hidden_sizes:
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            # Initialize the model, loss function, and optimizer
            model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=y_train.shape[1], dropout_rate=dropout_rate)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train the model
            num_epochs = 50  # Reduced for faster grid search
            for epoch in range(num_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                eval_outputs = model(X_eval)
                eval_loss = criterion(eval_outputs, y_eval)
                print(f'Hidden Size: {hidden_size}, Dropout Rate: {dropout_rate}, Learning Rate: {learning_rate}, Evaluation Loss: {eval_loss.item():.4f}')

                # Update best model if current model is better
                if eval_loss.item() < best_loss:
                    best_loss = eval_loss.item()
                    best_params = (hidden_size, dropout_rate, learning_rate)

print(f'Best Parameters: Hidden Size: {best_params[0]}, Dropout Rate: {best_params[1]}, Learning Rate: {best_params[2]}, Best Loss: {best_loss:.4f}')

# Train the best model with full epochs
best_model = LSTMModel(input_size=X_train.shape[2], hidden_size=best_params[0], output_size=y_train.shape[1], dropout_rate=best_params[1])
best_optimizer = optim.Adam(best_model.parameters(), lr=best_params[2])

# Train the best model
num_epochs = 100
for epoch in range(num_epochs):
    best_model.train()
    for X_batch, y_batch in train_loader:
        best_optimizer.zero_grad()
        outputs = best_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        best_optimizer.step()

# Predict changes with the best model
best_model.eval()
with torch.no_grad():
    predicted_changes = best_model(X_eval).numpy()

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