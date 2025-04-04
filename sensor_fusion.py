#!/usr/bin/env python3
import rospy
import numpy as np
import csv
import sys
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu, FluidPressure
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class SensorFusionNode:
    def __init__(self, imu_file, dvl_file, pressure_file, output_file):
        # 初始化传感器数据
        self.imu_data = []
        self.dvl_data = []
        self.pressure_data = []
        
        # 从CSV加载数据
        self.load_csv(imu_file, self.imu_data, ['timestamp', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                                               'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
        self.load_csv(dvl_file, self.dvl_data, ['timestamp', 'velocity_x', 'velocity_y', 'velocity_z'])
        self.load_csv(pressure_file, self.pressure_data, ['timestamp', 'fluid_pressure'])
        
        # 初始化轨迹和位置
        self.path = []
        self.last_time = None
        
        # 轨迹保存
        self.output_file = output_file
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'z'])
        
    def load_csv(self, file_path, data_list, headers):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = rospy.Time.from_sec(float(row['timestamp']))
                data = {k: float(v) if k != 'timestamp' else timestamp for k, v in row.items()}
                data_list.append(data)
    
    def fuse_sensors(self):
        if not self.imu_data or not self.dvl_data or not self.pressure_data:
            return
        
        # 按时间排序（假设输入数据已排序）
        self.imu_data.sort(key=lambda x: x['timestamp'])
        self.dvl_data.sort(key=lambda x: x['timestamp'])
        self.pressure_data.sort(key=lambda x: x['timestamp'])
        
        for imu, dvl, pressure in zip(self.imu_data, self.dvl_data, self.pressure_data):
            current_time = imu['timestamp']
            
            if self.last_time is not None and (current_time - self.last_time).to_sec() <= 0:
                continue
            
            # 从IMU获取姿态
            orientation = Quaternion(imu['orientation_x'], imu['orientation_y'], imu['orientation_z'], imu['orientation_w'])
            _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            
            # 从DVL获取速度
            vx_body = dvl['velocity_x']
            vy_body = dvl['velocity_y']
            
            # 转换到世界坐标系
            vx_world = vx_body * np.cos(yaw) - vy_body * np.sin(yaw)
            vy_world = vx_body * np.sin(yaw) + vy_body * np.cos(yaw)
            
            # 位置积分（简化模型，无加速度考虑）
            if self.last_time is None:
                dt = 0.1  # 假设固定时间步长
                self.position_x = 0.0
                self.position_y = 0.0
                self.position_z = 0.0
            else:
                dt = (current_time - self.last_time).to_sec()
            
            self.position_x += vx_world * dt
            self.position_y += vy_world * dt
            self.position_z = pressure['fluid_pressure'] / 1000.0  # 压力转深度（假设水密度1000kg/m³）
            
            # 保存轨迹点
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_time.to_sec(), self.position_x, self.position_y, self.position_z])
            
            self.last_time = current_time

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python script.py <imu_csv> <dvl_csv> <pressure_csv> <output_csv>")
        sys.exit(1)
    
    imu_file = sys.argv[1]
    dvl_file = sys.argv[2]
    pressure_file = sys.argv[3]
    output_file = sys.argv[4]
    
    node = SensorFusionNode(imu_file, dvl_file, pressure_file, output_file)
    
    # 模拟实时处理（实际非实时）
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        node.fuse_sensors()
        rate.sleep()