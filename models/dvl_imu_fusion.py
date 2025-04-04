import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
# from mamba.sylvester import Mamba
from mamba_ssm import Mamba
# import torchsummary

def visualize_poses(poses):
    # Example implementation
    plt.plot(poses)
    plt.show()

# =======================
# 1. 模拟数据生成器
# =======================
class SyntheticDataGenerator(Dataset):
    def __init__(self, seq_len=300, num_samples=100):
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 生成IMU数据（500Hz，6维）
        self.imu_data = []
        for _ in range(num_samples):
            # 位置：匀速直线运动 + 随机噪声
            pos = np.cumsum(np.random.randn(seq_len) * 0.1, axis=0) + np.random.uniform(0, 10, size=1)
            # 姿态：随机旋转 + 随机噪声
            attitude = np.random.uniform(-np.pi/2, np.pi/2, size=(seq_len, 3))  # roll, pitch, yaw
            # 合并IMU数据（加速度计+陀螺仪）
            imu = np.concatenate([
                pos[1:] - pos[:-1],          # 线速度（dx/dt）
                np.radians(attitude)[1:] - np.radians(attitude)[:-1]  # 角速度
            ], axis=1)
            self.imu_data.append(imu.astype(np.float32))
        
        # 生成DVL数据（7Hz，3维）
        self.dvl_data = []
        for _ in range(num_samples):
            # 速度：随机水平运动 + 垂直静止
            velocity = np.random.uniform(-0.5, 0.5, size=(seq_len//7)) * 7  # 转换为7Hz采样
            dvl = np.repeat(velocity, 7, axis=0)[:seq_len]  # 7Hz采样
            self.dvl_data.append(dvl.astype(np.float32))
        
        # 生成真实位姿（位置+姿态）
        self.true_poses = []
        for i in range(num_samples):
            pos = self.imu_data[i][:,0].cumsum()[:seq_len]
            attitude = np.cumsum(self.imu_data[i][:,1:4], axis=0)
            self.true_poses.append(np.concatenate([pos, attitude], axis=-1))

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        imu = torch.tensor(self.imu_data[idx], dtype=torch.float32).unsqueeze(0)
        dvl = torch.tensor(self.dvl_data[idx], dtype=torch.float32).unsqueeze(0)
        true_pose = torch.tensor(self.true_poses[idx], dtype=torch.float32).unsqueeze(0)
        return imu, dvl, true_pose

# =======================
# 2. 模型定义（混合架构）
# =======================

hidden_dim = 128

class IMUMambaEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super(IMUMambaEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.mamba = Mamba(
            d_model=hidden_dim,
            dt=0.1,
            max_dt=100,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            layer_norm=True
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Pass through fully connected layers
        x = self.fc(x)
        # Pass through Mamba
        mamba_output, _ = self.mamba(x)
        return mamba_output, torch.diag(torch.randn(self.hidden_dim))

# class IMUMambaEncoder(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=128):
#         super().__init__()
        
#         config = {
#             "d_model": hidden_dim,
#             "n_layers": 2,
#             "dt": 0.1,
#             "threshold": 100,
#             "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#             "layer_norm": True,
#             "rnn_mode": "lstm"
#         }
#         self.mamba = Mamba(**config) 
        
#     def forward(self, x):
#         return self.mamba(x)[0], torch.diag(torch.randn(hidden_dim))

class DVLTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_heads=4):
        super().__init__()
        self.pos_embed = nn.Embedding(hidden_dim, hidden_dim)(torch.arange(0, hidden_dim, dtype=torch.float32))
        self.encoder_layer = nn.TransformerEncoderLayer(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=num_heads,
            additive_pos_encoding=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        seq_len = x.shape[0]
        pos_encoding = self.pos_embed[:, None, :] * torch.arange(seq_len, dtype=torch.float32)[None, :, None]
        x = torch.cat([x, pos_encoding], dim=-1)
        return self.fc(self.encoder(x.permute(1, 0, 2))).squeeze(0), torch.diag(torch.randn(hidden_dim))

class FusionModule(nn.Module):
    def __init__(self, feat_dim_imu=128, feat_dim_dvl=64):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(feat_dim_imu + feat_dim_dvl, num_heads=8)
        self.fusion_fc = nn.Linear(feat_dim_imu + feat_dim_dvl + (feat_dim_imu + feat_dim_dvl)*8, 256)
        
    def forward(self, imu_feat, dvl_feat):
        # Cross-Attention
        attn_output, attn_weights = self.cross_attn(
            query=imu_feat.unsqueeze(1),
            key=dvl_feat.unsqueeze(1),
            value=dvl_feat.unsqueeze(1)
        )
        # Feature Fusion
        fused = torch.cat([
            imu_feat,
            attn_output.squeeze(1),
            torch.matmul(attn_weights.squeeze(1), dvl_feat)
        ], dim=-1)
        return self.fusion_fc(fused)

class UnderwaterSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.imu_encoder = IMUMambaEncoder(input_dim=6, hidden_dim=128)
        self.dvl_encoder = DVLTransformer(input_dim=3, hidden_dim=64, num_heads=4)
        self.fusion = FusionModule(128, 64)
        self.pose_pred = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        self.transition_gen = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 256),  # Transition Matrix A
            nn.Sigmoid()  # 输出0-1范围的矩阵元素
        )
        self.process_noise = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

    def forward(self, imu_seq, dvl_seq):
        # 特征提取
        imu_feat, _ = self.imu_encoder(imu_seq)
        dvl_feat, _ = self.dvl_encoder(dvl_seq)
        
        # 模态融合
        fused = self.fusion(imu_feat, dvl_feat)
        
        # 神经转移生成
        A = self.transition_gen(fused).view(256, 256)
        Q = torch.diag(torch.sigmoid(self.process_noise(fused))).squeeze()
        
        # 位姿预测
        return self.pose_pred(fused), A, Q

# =======================
# 3. 训练配置
# =======================
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# 初始化模型和优化器
model = UnderwaterSystem().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = amp.GradScaler()

# 数据加载器
dataset = SyntheticDataGenerator(seq_len=300, num_samples=100)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =======================
# 4. 训练循环
# =======================
writer = torch.utils.tensorboard.SummaryWriter(comment="Mamba_Convergence_Test")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    
    for imu, dvl, true_pose in dataloader:
        # 混合精度训练
        with amp.autocast():
            pred_pose, A, Q = model(imu.cuda(), dvl.cuda())
            loss = nn.MSELoss()(pred_pose.squeeze(), true_pose.squeeze())
            
        # 梯度计算与优化
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scaler.update()
        
        total_loss += loss.item() * imu.shape[0]
    
    # 验证
    model.eval()
    with torch.no_grad(), amp.autocast():
        val_loss = 0.0
        for imu, dvl, true_pose in dataloader:
            pred_pose, _, _ = model(imu.cuda(), dvl.cuda())
            val_loss += nn.MSELoss()(pred_pose.squeeze(), true_pose.squeeze()).item() * imu.shape[0]
        
    avg_train_loss = total_loss / len(dataset)
    avg_val_loss = val_loss / len(dataset)
    
    # 可视化
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_image('Poses', visualize_poses(true_pose[:10]), epoch, dataformats='HWC')
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

# =======================
# 5. 收敛性验证
# =======================
def visualize_poses(poses):
    # 将位姿矩阵转换为可视化形式
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(poses.shape[0]):
        pos = poses[i, :3]
        ax.scatter(pos[0], pos[1], pos[2], c=np.arange(poses.shape[0]), cmap='viridis')
        ax.plot([pos[0]], [pos[1]], [pos[2]], 'o-', markersize=2)
    
    return fig

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(writer.data['Loss/train'], label='Train Loss')
plt.plot(writer.data['Loss/val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# =======================
# 6. 推理性能测试
# =======================
def measure_inference_speed(model, dataloader):
    model.eval()
    start = torch.cuda.current_time()
    for imu, dvl, _ in dataloader:
        with torch.no_grad():
            model(imu.cuda(), dvl.cuda())
    latency = (torch.cuda.current_time() - start) / len(dataloader)
    return latency * 1000  # ms per batch

print(f"Inference Speed: {measure_inference_speed(model, dataloader):.2f} ms/batch")

# =======================
# 7. 模型分析
# =======================
# torchsummary.summary(model, input_size=(6, 300, 1), device="cuda")