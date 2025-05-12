import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

# Initialize ROS node
rospy.init_node('visual_inertial_odometry')

fx_l = 518.7188
fy_l = 518.7761
cx_l = 319.6680
cy_l = 234.1121
fx_r = 518.2156
fy_r = 518.4850
cx_r = 320.9156
cy_r = 234.2644
r11 = 9.99998555e-01
r12 = -2.98323440e-05
r13 = 1.69999383e-03
r21 = 2.94086111e-05
r22 = 9.99999968e-01
r23 = 2.49279992e-04
r31 = -1.70000121e-03
r32 = -2.49229637e-04
r33 = 9.99998524e-01
t1 = 5.01689369e-02
t2 = 1.31980135e-05
t3 = 4.07211339e-04

# Load camera parameters
intrinsic_left = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]])
intrinsic_right = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])
extrinsic = np.array([[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]])

# Initialize ROS publishers
path_pub = rospy.Publisher('/vehicle_path', Path, queue_size=10)
bridge = CvBridge()

# Initialize path message
path_msg = Path()
path_msg.header.frame_id = "world"

# Initialize variables for visual odometry
prev_left_img = None
prev_right_img = None
pose = np.eye(4)
imu_orientation = np.eye(3)

def left_image_callback(msg):
    global prev_left_img
    left_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    process_images(left_img, prev_right_img)
    prev_left_img = left_img

def right_image_callback(msg):
    global prev_right_img
    right_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    process_images(prev_left_img, right_img)
    prev_right_img = right_img

def imu_callback(msg):
    global imu_orientation
    orientation_q = msg.orientation
    imu_orientation = R.from_quat([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]).as_matrix()

def process_images(left_img, right_img):
    global pose, path_msg, path_pub, prev_left_img, prev_right_img, imu_orientation

    if left_img is None or right_img is None:
        return

    if prev_left_img is None or prev_right_img is None:
        prev_left_img = left_img
        prev_right_img = right_img
        return

    # Feature detection and matching
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_left_img, None)
    kp2, des2 = orb.detectAndCompute(left_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Estimate pose
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, intrinsic_left, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, intrinsic_left)

    # Update pose with IMU orientation
    pose[:3, :3] = imu_orientation @ R @ pose[:3, :3]
    pose[:3, 3] += t.flatten()

    # Create PoseStamped message
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "world"
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.pose.position.x = pose[0, 3]
    pose_msg.pose.position.y = pose[1, 3]
    pose_msg.pose.position.z = pose[2, 3]
    path_msg.poses.append(pose_msg)

    # Publish path
    path_pub.publish(path_msg)

# Subscribe to the image and IMU topics
rospy.Subscriber('/camera/infra1/image_rect_raw', Image, left_image_callback)
rospy.Subscriber('/camera/infra2/image_rect_raw', Image, right_image_callback)
rospy.Subscriber('/imu/data', Imu, imu_callback)

# Keep the script running
rospy.spin()
