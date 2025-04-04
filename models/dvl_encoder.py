import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from mamba_ssm import Mamba
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据读取模块
class DVLDataLoader(Dataset):
    def __init__(self, dvl_csv_path, gt_csv_path, dt=0.1):
        # 读取DVL数据
        self.dvl_df = pd.read_csv(dvl_csv_path)
        self.gt_df = pd.read_csv(gt_csv_path)
        
        # 时间同步预处理
        self.dvl_time = self.dvl_df['timestamp'].values / 1e9
        self.gt_time = self.gt_df['timestamp'].values / 1e9
        
        # 线性插值对齐到固定时间间隔dt
        self.sync_dvl = self.interpolate_data(self.dvl_df, self.dvl_time, dt)
        self.sync_gt = self.interpolate_data(self.gt_df, self.gt_time, dt)
        
        print(self.sync_gt.columns)

    def interpolate_data(self, df, time, dt):
        # 创建时间网格
        new_time = np.arange(time[0], time[-1], dt)
        
        # 插值函数
        interp_funcs = {
            'vx': lambda x: np.interp(x, time, df['vx']),
            'vy': lambda x: np.interp(x, time, df['vy']),
            'vz': lambda x: np.interp(x, time, df['vz'])
        }
        
        # 插值结果
        synced_data = {
            'timestamp': new_time,
            'vx': interp_funcs['vx'](new_time),
            'vy': interp_funcs['vy'](new_time),
            'vz': interp_funcs['vz'](new_time)
        }
        return pd.DataFrame(synced_data)
    
    def __len__(self):
        return len(self.sync_dvl)
    
    def __getitem__(self, idx):
        dvl_data = self.sync_dvl.iloc[idx][['vx', 'vy', 'vz']].values.astype(np.float32)
        gt_pose = self.sync_gt.iloc[idx][['vx', 'vy', 'vz']].values.astype(np.float32)
        return {
            'dvl_vel': torch.tensor(dvl_data),
            'gt_pose': torch.tensor(gt_pose)
        }

# 2. DVL编码器
class DVLEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # 返回特征和固定噪声协方差
        features = self.layers(x)
        noise_cov = torch.eye(features.shape[-1]) * 0.01
        return features, noise_cov

# 3. 动态建模（Mamba模型）
class DVLTransitionModel(nn.Module):
    def __init__(self, state_dim=6, d_model=128):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fc = nn.Linear(d_model, state_dim**2)  # 输出转移矩阵
    
    def forward(self, features):
        # features shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = features.shape
        state_dim = int(np.sqrt(self.fc.out_features))
        outputs = []
        
        # 逐时间步预测转移矩阵
        for t in range(seq_len):
            mamba_out = self.mamba(features[:, t, :].unsqueeze(1))
            A_flat = self.fc(mamba_out.squeeze(1))
            A = A_flat.view(batch_size, state_dim, state_dim)
            outputs.append(A)
        
        return torch.stack(outputs, dim=1)  # shape: (batch_size, seq_len, state_dim, state_dim)

# 4. 卡尔曼滤波器
class KalmanFilter:
    def __init__(self, state_dim=6, process_noise=0.01, measurement_noise=0.1):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # 初始化状态（位置+速度）
        self.x = torch.zeros(1, state_dim)
        self.P = torch.eye(state_dim) * 1e-3
        
        # 状态转移矩阵（恒速模型）
        self.A = torch.eye(state_dim)
        for i in range(3):
            self.A[i+3, i] = 1.0  # 速度项积分
        
        # 测量矩阵（仅观测速度）
        self.H = torch.zeros(state_dim, state_dim)
        self.H[3:6, 3:6] = torch.eye(3)
        
    def predict(self, A):
        # Ensure tensors are on the same device
        self.x = self.x.to(A.device)
        self.P = self.P.to(A.device)
        self.A = self.A.to(A.device)
        
        # 预测步骤
        self.x = torch.matmul(A, self.x.unsqueeze(-1)).squeeze(-1)
        self.P = torch.matmul(A, torch.matmul(self.P, A.transpose(-1, -2))) + \
                 self.process_noise * torch.eye(self.state_dim).to(A.device)
        
    def update(self, z):
        # Ensure tensors are on the same device
        self.H = self.H.to(z.device)
        self.P = self.P.to(z.device)
        
        # Extract the velocity components from the state vector
        x_vel = self.x[3:6]
        
        # 更新步骤
        K = torch.matmul(self.P, self.H.transpose(-1, -2)) / \
            (torch.matmul(self.H, torch.matmul(self.P, self.H.transpose(-1, -2))) + 
            self.measurement_noise)
        
        # Update only the velocity components
        x_vel = x_vel + torch.matmul(K[3:6, 3:6], z - torch.matmul(self.H[3:6, 3:6], x_vel.unsqueeze(-1))).squeeze(-1)
        
        # Update the full state vector
        self.x[3:6] = x_vel
        self.P = (torch.eye(self.state_dim).to(z.device) - torch.matmul(K, self.H)) @ self.P       

# 5. 主程序
def main():
    # 参数配置
    dvl_csv = 'dvl_davepool.csv'       # DVL原始数据路径
    gt_csv = 'gt_davepool.csv'     # 真实位姿数据路径
    dt = 0.1                       # 同步时间间隔(s)
    seq_length = 10                # 序列长度
    state_dim = 6                  # 状态维度
    
    # 数据加载
    dataset = DVLDataLoader(dvl_csv, gt_csv, dt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 模型初始化
    dvl_encoder = DVLEncoder()
    transition_model = DVLTransitionModel()
    kf = KalmanFilter()
    
    # 存储结果
    est_poses = []
    gt_poses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to GPU
    dvl_encoder.to(device)
    transition_model.to(device)

    # 推理循环
    with torch.no_grad():
        for batch in dataloader:
            dvl_vel = batch['dvl_vel'].unsqueeze(0).to(device)  # shape: (1, seq_len, 3)
            gt_pose = batch['gt_pose'].unsqueeze(0).to(device)  # shape: (1, seq_len, 3)
            
            # DVL编码器
            features, noise_cov = dvl_encoder(dvl_vel)
            
            # 动态建模
            A_sequence = transition_model(features)
            
            # 卡尔曼滤波
            for t in range(seq_length):
                # 预测
                if t == 0:
                    prev_A = torch.eye(state_dim).unsqueeze(0).to(device)
                else:
                    prev_A = A_sequence[:, t-1:t]
                kf.predict(prev_A)
                
                # 更新（假设每步都有测量）
                z = dvl_vel[:, t:t+1, :]  # shape: (1, 1, 3)
                kf.update(z)
                
                # 保存估计位姿
                est_pose = kf.x.squeeze().cpu().numpy()
                est_poses.append(est_pose)
                
            # 保存真实位姿
            gt_poses.extend(gt_pose.squeeze().cpu().numpy())
    
    # 转换为numpy数组
    est_poses = np.array(est_poses)
    gt_poses = np.array(gt_poses)
    
    # 绘制结果对比
    plt.figure(figsize=(12, 8))
    plt.suptitle("DVL-only Localization Evaluation")
    
    plt.subplot(3, 1, 1)
    plt.plot(gt_poses[:, 0], label='GT X')
    plt.plot(est_poses[:, 0], label='Est X')
    plt.title('X Position')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(gt_poses[:, 1], label='GT Y')
    plt.plot(est_poses[:, 1], label='Est Y')
    plt.title('Y Position')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(gt_poses[:, 2], label='GT Z')
    plt.plot(est_poses[:, 2], label='Est Z')
    plt.title('Z Position')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()