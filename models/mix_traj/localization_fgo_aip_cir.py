import gtsam
import gtsam.noiseModel
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.stats import chi2
import tf
from collections import deque

# Load the data from Transformer
transf_data = np.loadtxt('predicted_tum_cir_20250430_170250.txt')
transf_x = transf_data[:, 1]
transf_y = transf_data[:, 2]
transf_z = transf_data[:, 3]
transf_qx = transf_data[:, 4]
transf_qy = transf_data[:, 5]
transf_qz = transf_data[:, 6]
transf_qw = transf_data[:, 7]

# transf_x = transf_x - transf_x[0]
# transf_y = transf_y - transf_y[0]
# transf_z = transf_z - transf_z[0]

# Variance of the Transformer predictions
rel_transf_x = np.diff(transf_x)
rel_transf_y = np.diff(transf_y)
rel_transf_z = np.diff(transf_z)
var_rel_transf_x = np.var(rel_transf_x)
var_rel_transf_y = np.var(rel_transf_y)
# var_rel_transf_z = np.var(rel_transf_z)
var_transf_z = np.var(transf_z)

# print("Variance of Transformer relative motion (X):", var_rel_transf_x)
# print("Variance of Transformer relative motion (Y):", var_rel_transf_y)
# print("Variance of Transformer relative motion (Z):", var_rel_transf_z)

# Calculate quaternion differences between consecutive orientations
quat_diffs_transf = []
for i in range(1, len(transf_qw)):
    # Create scipy Rotation objects from quaternions
    r1 = R.from_quat([transf_qx[i-1], transf_qy[i-1], transf_qz[i-1], transf_qw[i-1]])
    r2 = R.from_quat([transf_qx[i], transf_qy[i], transf_qz[i], transf_qw[i]])

    # Get relative rotation
    r_diff = r1.inv() * r2
    
    # Extract euler angles (in radians)
    angles = r_diff.as_euler('xyz')
    quat_diffs_transf.append(angles)

quat_diffs_transf = np.array(quat_diffs_transf)

# Compute standard deviation for each rotation axis
roll_noise_transf = np.std(quat_diffs_transf[:, 0])   # Roll (x-axis)
pitch_noise_transf = np.std(quat_diffs_transf[:, 1])  # Pitch (y-axis)
yaw_noise_transf = np.std(quat_diffs_transf[:, 2])    # Yaw (z-axis)

# print(f"Roll noise: {roll_noise_transf:.4f} rad, Pitch noise: {pitch_noise_transf:.4f} rad, Yaw noise: {yaw_noise_transf:.4f} rad")

# Load the acoustic odometry data TBC
ao_data = np.loadtxt('dr_dvl_tum_cir_20250430_170250.txt')
ao_x = ao_data[:, 1]
ao_y = ao_data[:, 2]
ao_z = ao_data[:, 3]
ao_qx = ao_data[:, 4]
ao_qy = ao_data[:, 5]
ao_qz = ao_data[:, 6]
ao_qw = ao_data[:, 7]
# Variance of the acoustic odometry predictions
rel_ao_x = np.diff(ao_x)
rel_ao_y = np.diff(ao_y)
rel_ao_z = np.diff(ao_z)

var_rel_ao_x = np.var(rel_ao_x)
var_rel_ao_y = np.var(rel_ao_y)
var_rel_ao_z = np.var(rel_ao_z)

# print("Variance of Acoustic Odometry relative motion (X):", var_rel_ao_x)
# print("Variance of Acoustic Odometry relative motion (Y):", var_rel_ao_y)
# print("Variance of Acoustic Odometry relative motion (Z):", var_rel_ao_z)

# Calculate quaternion differences between consecutive orientations
quat_diffs_ao = []
for i in range(1, len(transf_qw)):
    # Create scipy Rotation objects from quaternions
    r1 = R.from_quat([ao_qx[i-1], ao_qy[i-1], ao_qz[i-1], ao_qw[i-1]])
    r2 = R.from_quat([ao_qx[i], ao_qy[i], ao_qz[i], ao_qw[i]])

    # Get relative rotation
    r_diff = r1.inv() * r2
    
    # Extract euler angles (in radians)
    angles = r_diff.as_euler('xyz')
    quat_diffs_ao.append(angles)

quat_diffs_ao = np.array(quat_diffs_ao)

# Compute standard deviation for each rotation axis
roll_noise_ao = np.std(quat_diffs_ao[:, 0])   # Roll (x-axis)
pitch_noise_ao = np.std(quat_diffs_ao[:, 1])  # Pitch (y-axis)
yaw_noise_ao = np.std(quat_diffs_ao[:, 2])    # Yaw (z-axis)

# ao_x = ao_x - ao_x[0]
# ao_y = ao_y - ao_y[0]
# ao_z = ao_z - ao_z[0]

# Factor graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# Add the initial estimate
for i in range(len(transf_qx)):
    initial_estimate.insert(i, gtsam.Pose3(gtsam.Rot3(transf_qx[i], transf_qy[i], transf_qz[i], transf_qw[i]),
                                           gtsam.Point3(transf_x[i], transf_y[i], transf_z[i])))

prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([var_rel_transf_x, var_rel_transf_y, var_transf_z,
              roll_noise_transf, pitch_noise_transf, yaw_noise_transf]))

# Add the first pose to the graph as a prior
graph.add(gtsam.PriorFactorPose3(0, initial_estimate.atPose3(0), prior_noise))

# Add the odometry factors
for i in range(1, len(transf_qx)):
    # Create poses for previous and current step
    prev_ao_pose = gtsam.Pose3(gtsam.Rot3.Quaternion(ao_qw[i-1], ao_qx[i-1], ao_qy[i-1], ao_qz[i-1]),
                               gtsam.Point3(ao_x[i-1], ao_y[i-1], ao_z[i-1]))
    
    curr_ao_pose = gtsam.Pose3(gtsam.Rot3.Quaternion(ao_qw[i], ao_qx[i], ao_qy[i], ao_qz[i]),
                               gtsam.Point3(ao_x[i], ao_y[i], ao_z[i]))
    
    # Calculate the relative transformation between consecutive poses
    relative_pose = prev_ao_pose.between(curr_ao_pose)
    
    # Create a noise model based on confidence in measurements
    # Lower values (e.g., 0.1) indicate higher confidence
    position_noise = 0.2  # Adjust based on your confidence in odometry
    rotation_noise = 0.1  # Adjust based on your confidence in odometry
    
    # Create a diagonal noise model with different values for position and rotation
    noise_model_ao = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([var_rel_ao_x, var_rel_ao_y, var_rel_ao_z,
                  roll_noise_ao, pitch_noise_ao, yaw_noise_ao])
    )
    
    # Add the between factor with the relative pose
    graph.add(gtsam.BetweenFactorPose3(i-1, i, relative_pose, noise_model_ao))

# Optimize the graph
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# Save the optimized poses
optimized_poses = []
for i in range(len(transf_qx)):
    pose = result.atPose3(i)
    optimized_poses.append(pose)

import matplotlib.pyplot as plt

# Load the ground truth data
gt_data = np.loadtxt('groundtruth_tum_cir_20250430_170250.txt')
gt_x = gt_data[:, 1]
gt_y = gt_data[:, 2]
gt_z = gt_data[:, 3]
gt_qx = gt_data[:, 4]
gt_qy = gt_data[:, 5]
gt_qz = gt_data[:, 6]
gt_qw = gt_data[:, 7]

# gt_x = gt_x - gt_x[0]
# gt_y = gt_y - gt_y[0]
# gt_z = gt_z - gt_z[0]

x_opt = [pose.x() for pose in optimized_poses]
y_opt = [pose.y() for pose in optimized_poses]
z_opt = [pose.z() for pose in optimized_poses]
# q_opt = [pose.rotation().quaternion() for pose in optimized_poses]
# q_opt = np.array(q_opt)
# qw_opt = q_opt[:, 0]
# qx_opt = q_opt[:, 1]
# qy_opt = q_opt[:, 2]
# qz_opt = q_opt[:, 3]

# Plot the optimized trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_opt, y_opt, label='Optimized Trajectory', color='blue')
plt.plot(gt_x, gt_y, label='Ground Truth', color='red', linestyle='--')
plt.scatter(gt_x[0], gt_y[0], color='green', label='Start (Ground Truth)', zorder=5)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Optimized Trajectory vs Ground Truth')
plt.legend()
plt.grid()
plt.show()

start_x, start_y = x_opt[0], y_opt[0]  # Start point of optimized trajectory

# 2. Create a rotation matrix for 180 degrees around z-axis
theta = np.radians(180)  # 180 degrees in radians
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 3. Translate, rotate, and translate back
x_opt_rotated = []
y_opt_rotated = []

for i in range(len(x_opt)):
    # Translate to origin
    x_centered = x_opt[i] - start_x
    y_centered = y_opt[i] - start_y
    
    # Apply rotation
    x_rotated, y_rotated = rotation_matrix @ np.array([x_centered, y_centered])
    
    # Translate back
    x_opt_rotated.append(x_rotated + start_x)
    y_opt_rotated.append(y_rotated + start_y)

# Now use x_opt_rotated and y_opt_rotated for plotting
plt.figure(figsize=(10, 6))
plt.plot(x_opt_rotated, y_opt_rotated, label='Optimized Trajectory (Rotated)', color='blue')
plt.plot(gt_x, gt_y, label='Ground Truth', color='red', linestyle='--')
plt.scatter(gt_x[0], gt_y[0], color='green', label='Start (Ground Truth)', zorder=5)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Rotated Optimized Trajectory vs Ground Truth')
plt.legend()
plt.grid()
plt.show()

# save the rotated optimized position and orientation to a txt file in tum format
with open('fgo_aip_transf_cir_20250430_170250_tum.txt', 'w') as f:
    for i, pose in enumerate(optimized_poses):
        timestamp = transf_data[i, 0]
        tx, ty, tz = pose.x(), pose.y(), pose.z()
        # Get the quaternion
        q = pose.rotation().toQuaternion()
        qw, qx, qy, qz = q.w(), q.x(), q.y(), q.z()
        
        # Write to file in TUM format
        f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

# debug
# plot the comparison of the trajectory provided by acoustic odometry and the ground truth
# plt.figure(figsize=(10, 6))
# plt.plot(ao_x, ao_y, label='Acoustic Odometry', color='orange')
# plt.plot(gt_x, gt_y, label='Ground Truth', color='red', linestyle='--')
# plt.scatter(gt_x[0], gt_y[0], color='green', label='Start (Ground Truth)', zorder=5)
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.title('Acoustic Odometry vs Ground Truth')
# plt.legend()
# plt.grid()
# plt.show()