#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class StereoVisualOdometry:
    def __init__(self):
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None
        self.prev_left_image = None
        self.prev_right_image = None
        self.pose = np.eye(4)  # Initial pose is the identity matrix
        self.path = Path()  # Initialize the path

        # Camera parameters
        self.left_camera_matrix = np.array([[518.7188, 0, 319.6680],
                                            [0, 518.7761, 234.1121],
                                            [0, 0, 1]])
        self.right_camera_matrix = np.array([[518.2156, 0, 320.9156],
                                             [0, 518.4850, 234.2644],
                                             [0, 0, 1]])
        self.left_dist_coeffs = np.array([0.2154, 0.3974, 0, 0])
        self.right_dist_coeffs = np.array([0.1928, 0.4041, 0, 0])
        self.R = np.array([[9.99998555e-01, -2.98323440e-05, 1.69999383e-03],
                           [2.94086111e-05, 9.99999968e-01, 2.49279992e-04],
                           [-1.70000121e-03, -2.49229637e-04, 9.99998524e-01]])
        self.T = np.array([5.01689369e-02, 1.31980135e-05, 4.07211339e-04])

        # Subscribers
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.left_callback)
        rospy.Subscriber("/camera/infra2/image_rect_raw", Image, self.right_callback)

        # TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Pose publisher
        self.pose_pub = rospy.Publisher("/vehicle_pose", PoseStamped, queue_size=10)

        # Path publisher
        self.path_pub = rospy.Publisher("/vehicle_path", Path, queue_size=10)

    def left_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def right_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def rectify_images(self, left_image, right_image):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            left_image.shape[:2], self.R, self.T
        )

        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs, R1, P1, left_image.shape[:2], cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs, R2, P2, right_image.shape[:2], cv2.CV_16SC2
        )

        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        return left_rectified, right_rectified

    def detect_and_match_features(self, left_image, right_image, prev_left_image, prev_right_image):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(left_image, None)
        kp2, des2 = orb.detectAndCompute(right_image, None)
        kp3, des3 = orb.detectAndCompute(prev_left_image, None)
        kp4, des4 = orb.detectAndCompute(prev_right_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_left = bf.match(des1, des3)
        matches_right = bf.match(des2, des4)

        matches_left = sorted(matches_left, key=lambda x: x.distance)
        matches_right = sorted(matches_right, key=lambda x: x.distance)

        return matches_left, matches_right, kp1, kp2, kp3, kp4

    def estimate_pose(self, matches_left, matches_right, kp1, kp2, kp3, kp4):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches_left])
        pts2 = np.float32([kp3[m.trainIdx].pt for m in matches_left])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.left_camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.left_camera_matrix)

        # Update pose
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t.flatten()
        self.pose = np.dot(self.pose, np.linalg.inv(transformation))

        # Print pose information
        print("Estimated Pose:\n", self.pose)

        # Broadcast the pose
        self.broadcast_pose()

        # Publish the pose
        self.publish_pose()

        # Update and publish the path
        self.update_path()

    def broadcast_pose(self):
        translation = self.pose[:3, 3]
        rotation = quaternion_from_matrix(self.pose)
        self.tf_broadcaster.sendTransform(translation, rotation, rospy.Time.now(), "vehicle", "world")

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = self.pose[0, 3]
        pose_msg.pose.position.y = self.pose[1, 3]
        pose_msg.pose.position.z = self.pose[2, 3]
        q = quaternion_from_matrix(self.pose)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        self.pose_pub.publish(pose_msg)

    def update_path(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = self.pose[0, 3]
        pose_msg.pose.position.y = self.pose[1, 3]
        pose_msg.pose.position.z = self.pose[2, 3]
        q = quaternion_from_matrix(self.pose)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        # Append the current pose to the path
        self.path.header = pose_msg.header
        self.path.poses.append(pose_msg)
        self.path_pub.publish(self.path)

    def process_images(self):
        if self.left_image is not None and self.right_image is not None:
            if self.prev_left_image is not None and self.prev_right_image is not None:
                left_rectified, right_rectified = self.rectify_images(self.left_image, self.right_image)
                prev_left_rectified, prev_right_rectified = self.rectify_images(self.prev_left_image, self.prev_right_image)

                matches_left, matches_right, kp1, kp2, kp3, kp4 = self.detect_and_match_features(
                    left_rectified, right_rectified, prev_left_rectified, prev_right_rectified
                )

                self.estimate_pose(matches_left, matches_right, kp1, kp2, kp3, kp4)

            self.prev_left_image = self.left_image
            self.prev_right_image = self.right_image

if __name__ == "__main__":
    rospy.init_node('stereo_visual_odometry')
    vo = StereoVisualOdometry()
    rospy.spin()