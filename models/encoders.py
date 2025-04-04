import torch
from torch import nn

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),  # Layer1
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # Layer2
            nn.LeakyReLU(),
            # ...（用户提供的各层结构）
        )
        self.fc_feature = nn.Linear(128, 64)    # 输出视觉特征
        self.fc_noise = nn.Linear(128, 64)      # 输出视觉噪声协方差
    def forward(self, images):
        x = self.conv_layers(images)
        a_v = self.fc_feature(x)    # B×64
        R_v = torch.exp(self.fc_noise(x))  # 保证协方差正定
        return a_v, R_v
    
class InertialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, num_layers=2, batch_first=True)
        self.fc_feature = nn.Linear(32, 64)     # 输出惯性特征
        self.fc_noise = nn.Linear(32, 64)       # 输出惯性噪声协方差
    def forward(self, imu):
        x, _ = self.lstm(imu)
        a_i = self.fc_feature(x[:, -1, :])      # B×64
        R_i = torch.exp(self.fc_noise(x[:, -1, :]))
        return a_i, R_i
    
class DVLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
        self.fc_feature = nn.Linear(16, 64)     # 输出声学特征
        self.fc_noise = nn.Linear(16, 64)       # 输出声学噪声协方差
    def forward(self, dvl):
        x, _ = self.lstm(dvl)
        a_d = self.fc_feature(x[:, -1, :])      # B×64
        R_d = torch.exp(self.fc_noise(x[:, -1, :]))
        return a_d, R_d
    
class PressureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc_feature = nn.Linear(32, 64)     # 输出深度特征
        self.fc_noise = nn.Linear(32, 64)       # 输出压力噪声协方差
    def forward(self, pressure):
        x = F.relu(self.fc1(pressure[:, -1, :]))  # 取最新压力值
        z_p = self.fc_feature(x)               # B×64（深度与垂直速度）
        R_p = torch.exp(self.fc_noise(x))
        return z_p, R_p

class SensorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(4))  # 可学习权重
    def forward(self, a_v, R_v, a_i, R_i, a_d, R_d, z_p, R_p):
        # 计算各传感器权重
        w = F.softmax(self.weights, dim=0)  
        # 加权融合特征
        a_fused = w[0]*a_v + w[1]*a_i + w[2]*a_d + w[3]*z_p
        # 融合噪声协方差
        R_fused = w[0]*R_v + w[1]*R_i + w[2]*R_d + w[3]*R_p
        return a_fused, R_fused

class DynamicModel(nn.Module):
    def __init__(self, state_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=128, batch_first=True)
        self.alpha = nn.Parameter(torch.ones(state_dim))  # Dirichlet参数
    def forward(self, z_prev):
        # 生成转移矩阵A_t
        A_t = torch.distributions.dirichlet.Dirichlet(self.alpha).rsample()  # B×128
        # 生成过程噪声Q_t
        h, _ = self.lstm(z_prev)
        Q_t = torch.exp(h[:, -1, :])  # B×128
        return A_t, Q_t
    
class DifferentiableKF(nn.Module):
    def forward(self, z_prev, P_prev, A, Q, a_obs, R_obs):
        # 预测步
        z_pred = A @ z_prev
        P_pred = A @ P_prev @ A.T + Q
        # 更新步
        K = P_pred @ torch.inverse(P_pred + R_obs)
        z_curr = z_pred + K @ (a_obs - z_pred)
        P_curr = (torch.eye(128) - K) @ P_pred
        return z_curr, P_curr

class FusionKF(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_enc = VisualEncoder()
        self.imu_enc = InertialEncoder()
        self.dvl_enc = DVLEncoder()      
        self.pressure_enc = PressureEncoder()
        self.fusion = SensorFusion()
        self.dynamic_model = DynamicModel()
        self.kf = DifferentiableKF()
        self.pose_predictor = nn.Linear(128, 6)

    def forward(self, batch):
        a_v, R_v = self.visual_enc(batch['camera'])
        a_i, R_i = self.imu_enc(batch['imu'])
        a_d, R_d = self.dvl_enc(batch['dvl'])
        z_p, R_p = self.pressure_enc(batch['pressure'])
        
        a_fused, R_fused = self.fusion(a_v, R_v, a_i, R_i, a_d, R_d, z_p, R_p)
        A, Q = self.dynamic_model(z_prev)
        z_curr, P_curr = self.kf(z_prev, P_prev, A, Q, a_fused, R_fused)
        poses = self.pose_predictor(z_curr)
        return poses
