import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UnderwaterDataset(Dataset):
    def __init__(self, data_dir, seq_length=10):
        self.data_dir = data_dir
        self.seq_length = seq_length
        # 加载各传感器CSV
        self.cam_data = pd.read_csv(f"{data_dir}/camera.csv")
        self.imu_data = pd.read_csv(f"{data_dir}/imu.csv")
        self.dvl_data = pd.read_csv(f"{data_dir}/dvl.csv")
        self.pressure_data = pd.read_csv(f"{data_dir}/pressure.csv")
        self.gt_poses = pd.read_csv(f"{data_dir}/gt_poses.csv")
        
        # 时间同步（线性插值）
        self._sync_data()

    def _sync_data(self):
        """将视觉、DVL、压力数据插值到IMU时间戳"""
        t_imu = self.imu_data['timestamp'].values
        self.cam_synced = self._interpolate(self.cam_data, t_imu)
        self.dvl_synced = self._interpolate(self.dvl_data, t_imu)
        self.pressure_synced = self._interpolate(self.pressure_data, t_imu)

    def _interpolate(self, df, t_target):
        return np.interp(t_target, df['timestamp'], df['data'])

    def __len__(self):
        return len(self.imu_data) - self.seq_length

    def __getitem__(self, idx):
        # 提取序列数据
        imu_seq = self.imu_data.iloc[idx:idx+self.seq_length, 1:7].values  # 6维：acc+gyro
        cam_seq = self.cam_synced[idx:idx+self.seq_length]               # 图像路径
        dvl_seq = self.dvl_synced[idx:idx+self.seq_length, 1:4].values     # 3维速度
        pressure_seq = self.pressure_synced[idx:idx+self.seq_length]
        gt_pose = self.gt_poses.iloc[idx+self.seq_length, 1:7].values     # 6自由度位姿
        
        # 转换为Tensor
        return {
            'imu': torch.FloatTensor(imu_seq),
            'camera': self._load_image(cam_seq),
            'dvl': torch.FloatTensor(dvl_seq),
            'pressure': torch.FloatTensor(pressure_seq),
            'pose_gt': torch.FloatTensor(gt_pose)
        }

    def _load_image(self, paths):
        # 加载并预处理图像（示例）
        images = [cv2.imread(p) for p in paths]
        return torch.stack([torch.FloatTensor(img) for img in images])