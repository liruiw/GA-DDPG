#!/usr/bin/env python

# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# for testing    
import argparse
import datetime
 
import numpy as np
import itertools
from core.bc import BC
from core.ddpg import DDPG
from tensorboardX import SummaryWriter
 
from experiments.config import * 
from core.replay_memory import BaseMemory as ReplayMemory
from core import networks
from core.utils import *
import IPython
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import cv2
import torch.nn as nn
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import copy
from core.env_planner import EnvPlanner
from OMG.omg.config import cfg as planner_cfg 

# try: # ros
import tf
import tf2_ros
import rosnode
import message_filters    
import _init_paths
import rospy
import tf.transformations as tra

import std_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
lock = threading.Lock()

# for real robot
from lula_franka.franka import Franka
from joint_listener import JointListener
from moveit import MoveitBridge

# use posecnn layer for backprojection
import posecnn_cuda

# graspnet
import tensorflow
sys.path.insert(0, '6dof-graspnet')

# set policy mode
GA_DDPG_ONLY = True
GRASPNET_ONLY = False
COMBINED = False
RANDOM_TARGET = False
USE_LOOK_AT = False
CONTACT_GRASPNET = False
PUT_BIN = False

# contact graspnet
from grasp_estimator import GraspEstimator, get_graspnet_config, joint_config
if CONTACT_GRASPNET:
    sys.path.insert(0, 'contact_graspnet')
    sys.path.insert(0, 'contact_graspnet/contact_graspnet')
    from inference_edit import get_graspnet_config as get_graspnet_config_contact
    from contact_grasp_estimator import GraspEstimator as GraspEstimatorContact
    import config_utils


# compute look at pose according to object pose
def compute_look_at_pose(pose_listener, center_object, angle, distance, psi=0):

    # find the hand camera to hand transformation
    try:
        tf_pose = pose_listener.lookupTransform('measured/camera_color_optical_frame', 'measured/right_gripper', rospy.Time(0))
        pose_camera = make_pose(tf_pose)
    except (tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException):
        pose_camera = None

    if pose_camera is not None:
        pose_camera[:3, :3] = np.eye(3)
        pose_camera[:3, 3] *= -1
    else:
        print('cannot find camera to hand transformation')

    psi /= 57.3
    theta = angle / 57.3
    r = distance
    position_robot = center_object + np.array([-r * np.cos(theta) * np.cos(psi),
                                               -r * np.cos(theta) * np.sin(psi),
                                                r * np.sin(theta)], dtype=np.float32)
    Z_BG = center_object - position_robot
    Z_BG /= np.linalg.norm(Z_BG)
    Y_BG = np.array([-np.sin(psi), np.cos(psi), 0], dtype=np.float32)
    X_BG = np.cross(Y_BG, Z_BG)
    R_BG = np.zeros((3, 3), dtype=np.float32)
    R_BG[:, 0] = X_BG
    R_BG[:, 1] = Y_BG
    R_BG[:, 2] = Z_BG

    pose_robot = np.eye(4, dtype=np.float32)
    pose_robot[:3, 3] = position_robot
    pose_robot[:3, :3] = R_BG[:3, :3]

    # adjust for camera offset
    if pose_camera is not None:
        pose_robot = np.dot(pose_camera, pose_robot)
    return pose_robot


class ImageListener:

    def __init__(self, agent, graspnet, graspnet_contact):

        franka = Franka(is_physical_robot=True)
        self.moveit = MoveitBridge(franka)
        self.moveit.retract()

        # self.moveit.close_gripper()
        self.moveit.open_gripper()

        self.joint_listener = JointListener()
        self.pose_listener = tf.TransformListener()
        print('sleep a short time')
        rospy.sleep(2.0)
        print('current robot joints')
        print(self.joint_listener.joint_position)

        tf_pose = self.pose_listener.lookupTransform('measured/panda_hand', 'measured/right_gripper', rospy.Time(0))
        self.grasp_offset = make_pose(tf_pose)
        print('grasp offset', self.grasp_offset)

        self.agent = agent
        self.graspnet = graspnet
        self.graspnet_contact = graspnet_contact
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.im_ef_pose = None
        self.acc_points = np.zeros([4, 0])
        self.depth_threshold = 1.2
        self.table_height = 0.0
        self.initial_joints = initial_joints
        self.num_initial_joints = initial_joints.shape[0]
        self.index_joints = 0
        self.target_obj_id = 1 # target object ID

        # publish object points for visualization
        self.empty_msg = PointCloud2()
        self.object_points2_target_pub = rospy.Publisher('/gaddpg_object_points2_target', PointCloud2, queue_size=10)
        self.object_points2_obstacle_pub = rospy.Publisher('/gaddpg_object_points2_obstacle', PointCloud2, queue_size=10)

        # initialize a node 
        self.label_sub = message_filters.Subscriber('seg_label', Image, queue_size=1)

        self.hand_finger_point = np.array([ [ 0.,  0.,  0.   , -0.   ,  0.   , -0.   ],
                               [ 0.,  0.,  0.053, -0.053,  0.053, -0.053],
                               [ 0.,  0.,  0.075,  0.075,  0.105,  0.105]])

        self.bin_conf_1 = np.array([0.7074745589850109, 0.361727706885124, 0.38521270434333, 
            -1.1754794559646125, -0.4169872830046795, 1.7096866963969337, 1.654512471818922]).astype(np.float32)

        self.bin_conf_2 = np.array([0.5919747534674433, 0.7818432665691674, 0.557417382701195, 
            -1.1647884021323738, -0.39191044586242046, 1.837464805311654, 1.9150514982533562]).astype(np.float32)

        if cfg.ROS_CAMERA == 'D415':
            # use RealSense D435
            self.base_frame = 'measured/base_link'
            camera_name = 'cam_2'
            rgb_sub = message_filters.Subscriber('/%s/color/image_raw' % camera_name, Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/aligned_depth_to_color/image_raw' % camera_name, Image, queue_size=1)
            msg = rospy.wait_for_message('/%s/color/camera_info' % camera_name, CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif cfg.ROS_CAMERA == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (cfg.ROS_CAMERA)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.ROS_CAMERA), Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.ROS_CAMERA), Image, queue_size=1)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.ROS_CAMERA), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (cfg.ROS_CAMERA)
            self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.4
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, self.label_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbdm)

        # set global intrinsics and extrinsics 
        global INTRINSICS, EXTRINSICS
        INTRINSICS = intrinsics
        EXTRINSICS = np.zeros([4, 4])# from camera to end effector
        EXTRINSICS[:3, 3]  = (np.array([0.05253322227958818, -0.05414890498307623, 0.06035263861136299]))   # camera offset
        EXTRINSICS[:3, :3] = quat2mat([0.7182116422267757, 0.016333297635292354, 0.010996322012974747, 0.6955460741463947])
        self.remaining_step = cfg.RL_MAX_STEP

        # start publishing thread
        self.start_publishing_tf()
        self.planner = EnvPlanner()
        self.expert_plan = []
        self.standoff_idx = -1
        self.has_plan = False
        self.num_trial = 0
        # threshold to close gripper
        self.grasp_score_threshold = 0.4


    def compute_plan_with_gaddpg(self, state, ef_pose, vis=False):
        """
        generate initial expert plan
        """
        joints = get_joints(self.joint_listener)
        gaddpg_grasps_from_simulate_view(self.agent, state, cfg.RL_MAX_STEP, ef_pose)
        print('finish simulate views')
        # can use remaining timesteps to replan. Set vis to visualize collision and traj
        self.expert_plan, self.standoff_idx = self.planner.expert_plan(cfg.RL_MAX_STEP, joints, ef_pose, state[0][0], vis=vis)
        print('expert plan', self.expert_plan.shape)
        print('standoff idx', self.standoff_idx)

        
    def start_publishing_tf(self):
        self.stop_event = threading.Event()
        self.tf_thread = threading.Thread(target=self.publish_point_cloud)
        self.tf_thread.start()


    def publish_point_cloud(self):
        rate = rospy.Rate(30.)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.base_frame
            out_xyz = self.acc_points[:3, :].T
            label = self.acc_points[3, :].flatten()

            target_xyz = out_xyz[label == 0, :]
            obj_pc2_target = point_cloud2.create_cloud(header, fields, target_xyz)
            self.object_points2_target_pub.publish(obj_pc2_target)

            obstacle_xyz = out_xyz[label == 1, :]
            obj_pc2_obstacle = point_cloud2.create_cloud(header, fields, obstacle_xyz)
            self.object_points2_obstacle_pub.publish(obj_pc2_obstacle)

            # if out_xyz.shape[0] > 0:
            #     print('publish points')
            #     print(out_xyz.shape)
            rate.sleep()


    def callback_rgbdm(self, rgb, depth, mask):

        ef_pose = get_ef_pose(self.pose_listener)
        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        mask = self.cv_bridge.imgmsg_to_cv2(mask, 'mono8')

        # rescale image if necessary
        # Lirui: consider rescaling to 112 x 112 which is used in training (probably not necessary)
        if cfg.SCALES_BASE[0] != 1:
            im_scale = cfg.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)
            mask = pad_im(cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)
        
        with lock:
            self.im = im.copy()
            self.im_ef_pose = ef_pose.copy()
            self.mask = mask.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def show_segmentation_result(self, color, mask, mask_ids):

        image = color.copy()
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            index = np.where(mask == mask_id)
            x = int(np.mean(index[1]))
            y = int(np.mean(index[0]))
            image = cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.namedWindow("Display 1")
        cv2.imshow("Display 1", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        value = input('Please enter which object to pick up: ')
        return int(value)


    def find_target_object(self, depth, mask, mask_ids, ef_pose, remaining_step, vis=False):

        # select target points
        target_mask = get_target_mask(self.acc_points)
        points = self.acc_points[:3, target_mask]

        # sample points
        points = regularize_pc_point_count(points.T, 1024, use_farthest_point=True).T

        # base to hand
        points = se3_transform_pc(se3_inverse(ef_pose), points)

        # hand to camera
        offset_pose = se3_inverse(EXTRINSICS)
        xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]

        # projection to image
        p_xyz = INTRINSICS.dot(xyz_points)
        index = p_xyz[2] > 0.03
        p_xyz = p_xyz[:, index]
        xyz_points = xyz_points[:, index]
        x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)

        # bounding box
        x1 = np.min(x)
        x2 = np.max(x)
        y1 = np.min(y)
        y2 = np.max(y)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # check labels
        valid_idx_mask = (x > 0) * (x < mask.shape[1] - 1) * (y > 0) * (y < mask.shape[0] - 1)
        labels = mask[y[valid_idx_mask], x[valid_idx_mask]]
        labels_nonzero = labels[labels > 0]
        xyz_points = xyz_points[:, valid_idx_mask]

        # find the marjority label
        if float(len(labels_nonzero)) / float((len(labels) + 1)) < 0.5:
            print('overlap to background')
            target_id = -1
        else:
            target_id = np.bincount(labels_nonzero).argmax()

            # check bounding box overlap
            I = np.where(mask == target_id)
            x11 = np.min(I[1])
            x22 = np.max(I[1])
            y11 = np.min(I[0])
            y22 = np.max(I[0])
            area1 = (x22 - x11 + 1) * (y22 - y11 + 1)

            xx1 = np.maximum(x1, x11)
            yy1 = np.maximum(y1, y11)
            xx2 = np.minimum(x2, x22)
            yy2 = np.minimum(y2, y22)

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (area + area1 - inter)
            print('overlap', ovr)
            if ovr < 0.3:
                target_id = -1

            # projected depth
            depths = depth[y[valid_idx_mask], x[valid_idx_mask]]
            # computed depth
            z = xyz_points[2, :]
            diff = np.mean(np.absolute(depths - z))
            print('mean depth diff', diff)
            if diff > 0.15:
                target_id = -1

        # if remaining_step == cfg.RL_MAX_STEP - 1 and target_id != -1:
        #    self.acc_points = np.zeros([4, 0])

        if vis:
            # show image
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(mask)
            plt.scatter(x[valid_idx_mask], y[valid_idx_mask], s=10)
            # plt.show()
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        return target_id


    def print_joint(self, joint):
        num = len(joint)
        s = ''
        for i in range(num):
            s += '%.6f, ' % rad2deg(joint[i])
        print(s)


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def compute_grasp_object_distance(self, RT_grasp):
        T = RT_grasp[:3, 3].reshape((3, 1))

        # target points
        index = self.acc_points[3, :] == 0
        points = self.acc_points[:3, index]
        n = points.shape[1]

        hand = np.repeat(T, n, axis=1)
        distances = np.linalg.norm(hand - points, axis=0)
        return np.min(distances)


    def run_network(self):

        # sample an initial joint
        if self.remaining_step == cfg.RL_MAX_STEP:
            print('use initial joint %d' % (self.index_joints))
            initial_joints = self.initial_joints[self.index_joints, :]
            self.moveit.go_local(q=initial_joints, wait=True)
            rospy.sleep(1.0)

        with lock:
            if listener.im is None:
                print('no image')
                return
            color = self.im.copy()
            depth = self.depth.copy()
            mask = self.mask.copy()
            im_ef_pose = self.im_ef_pose.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')

        # process mask
        mask = self.process_label(mask)
        mask_ids = np.unique(mask)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = mask_ids.shape[0]
        mask_failure = (num == 0)

        # no mask for the first frame
        if mask_failure and self.remaining_step == cfg.RL_MAX_STEP:
            print('no object segmented')
            raw_input('put objects in the scene?')
            return

        count = np.zeros((num, ), dtype=np.int32)
        for i in range(num):
            count[i] = len(np.where(mask == mask_ids[i])[0])

        # show the segmentation
        start_time = time.time()
        if self.remaining_step == cfg.RL_MAX_STEP:
            print('%d objects segmented' % num)
            print(mask_ids)


            if not RANDOM_TARGET:
                label_max = np.argmax(count)
                target_id = mask_ids[label_max]
            else:
                target_id = self.show_segmentation_result(color, mask, mask_ids)
                '''
                while True:
                    target_id = np.random.choice(mask_ids)
                    # check number of pixels for the target
                    num_pixels = np.sum(mask == target_id)
                    if num_pixels > 500:
                        print('%d target pixels' % num_pixels)
                        break
                '''
        elif num > 0:
            # data association to find the target id for the current frame
            target_id = self.find_target_object(depth, mask, mask_ids, im_ef_pose, self.remaining_step, vis=False)
        else:
            target_id = -1
        self.target_obj_id = target_id
        print('target id is %d' % target_id)
        print("---select target time %s seconds ---" % (time.time() - start_time))

        if self.remaining_step == cfg.RL_MAX_STEP and not args.fix_initial_state:
            self.index_joints += 1
            if self.index_joints >= self.num_initial_joints:
                self.index_joints = 0

        # process target mask
        start_time = time.time()
        mask_background = np.zeros_like(mask)
        mask_background[mask == 0] = 1
        if num > 0:
            # update this for 0 background and 1-N for other target
            mask_target = np.zeros_like(mask)
            mask_target[mask == target_id] = 1
            # erode target mask
            mask_target = cv2.erode(mask_target, np.ones((7, 7), np.uint8), iterations=3)
            num_pixels = np.sum(mask_target)
            print('finish mask, %d foreground pixels' % (num_pixels))
            # build the final mask
            mask[(mask == target_id) & (mask_target == 0)] = 0
            mask_final = mask.copy()
        else:
            mask_final = np.zeros_like(mask)
        print("---process mask time %s seconds ---" % (time.time() - start_time))

        # compute state
        start_time = time.time()
        depth = depth[...,None]
        agg = (not mask_failure) and (self.remaining_step >= cfg.RL_MAX_STEP - 1)
        state, point_background = self.camera_image_to_state( color, depth, mask_final, mask_background, im_ef_pose,
                                            cfg.RL_MAX_STEP - self.remaining_step, 
                                            agg=agg, vis=False)
        print('after camera image to state', state[0].shape)
        print('background point shape', point_background.shape)
        print("---compute state time %s seconds ---" % (time.time() - start_time))

        # compute action
        state = [state, None, None, None]

        # look at target
        if self.remaining_step == cfg.RL_MAX_STEP and USE_LOOK_AT:
            index = self.acc_points[3, :] == 0
            points = self.acc_points[:3, index]
            center = np.mean(points, axis=1)
            angle = 60
            T_lookat = compute_look_at_pose(self.pose_listener, center, angle=angle, distance=0.45)
            self.moveit.go_local(T_lookat, wait=True)
            self.remaining_step = max(self.remaining_step-1, 1)
            rospy.sleep(0.5)
            return

        # GRASPNET + OMG + GA-DDPG

        # run graspnet
        if (not self.has_plan and COMBINED) or (GRASPNET_ONLY and not GA_DDPG_ONLY):
            point_state = state[0][0].copy() # avoid aggregation
            print('point_state', point_state.shape)
            target_mask = point_state[3, :] == 0        
            target_pt = point_state[:3, target_mask].T
            print('target_pt', target_pt.shape)

            if CONTACT_GRASPNET: # only for target
                #  pc_full: (493949, 3), pc_colors: (493949, 3), pc_segments: dict (idx: (13481, 3)), local_regions True filter_grasps True forward_passes 1
                pc_segments = {'0': target_pt}
                point_full = point_state[:3,6:-500].T
                print('point_full', point_full.shape)
                # all points. You need to add table point here
                pred_grasps_cam, scores, contact_pts, _ = self.graspnet_contact.predict_scene_grasps(sess_contact, point_full, 
                                                                                             pc_segments=pc_segments, 
                                                                                             local_regions=True,
                                                                                             filter_grasps=True, 
                                                                                             forward_passes=1)  
                # pred_grasps_cam: dict (idx: (N, 4, 4)), scores: dict (idx: (N, 1)), contact_pts: dict (idx: (N, 3))
                generated_grasps = pred_grasps_cam['0']
                generated_scores = scores['0']
                print('generated contact grasps', generated_grasps.shape)
            else:
                latents = self.graspnet.sample_latents()
                generated_grasps, generated_scores, _ = self.graspnet.predict_grasps(
                    sess,
                    target_pt.copy(),
                    latents,
                    num_refine_steps=10,
                )

            # select grasps
            top_num = 100 # grasp num
            sorted_idx = list(np.argsort(generated_scores))[::-1]
            select_grasp  = [generated_grasps[idx] for idx in sorted_idx[:top_num]]  
            select_grasp_score = [generated_scores[idx] for idx in sorted_idx[:top_num]]
            print('mean select grasp score: {:.3f}'.format(np.mean(np.round(select_grasp_score, 3))))
            goal_states = np.array([im_ef_pose.dot(g.dot(rotZ(np.pi / 2))) for g in select_grasp]) # might not need rotate
            print(goal_states.shape)
            if goal_states.shape[0] == 0:
                return

            # use OMG in this repo
            planner_cfg.use_external_grasp = True
            planner_cfg.external_grasps = goal_states # this sets the grasps in base coordinate
            joints = get_joints(self.joint_listener)

            # construct scene points
            num = point_state.shape[1] + point_background.shape[1]
            scene_points = np.ones((4, num), dtype=np.float32)
            scene_points[:, :point_state.shape[1]] = point_state.copy()
            scene_points[:3, point_state.shape[1]:] = point_background.copy()

            step = 30
            plan, standoff_idx = self.planner.expert_plan(step, joints, im_ef_pose, scene_points, vis=False)
            self.has_plan = True
            print('expert plan', plan.shape)

            # execute plan to standoff
            if COMBINED:
                self.moveit.execute(plan[:standoff_idx-5])
                self.remaining_step = 10
                print('*****************switch to gaddpg****************')
                rospy.sleep(1.0)
            else:
                self.moveit.execute(plan[:standoff_idx])
                self.moveit.execute(plan[standoff_idx:])
                rospy.sleep(1.0)
                if PUT_BIN:
                    self.put_bin()
                else:
                    self.retract()
                self.acc_points = np.zeros([4, 0])
                self.remaining_step = cfg.RL_MAX_STEP
        else:

            if self.termination_heuristics(state) or self.num_trial >= 5:
                if self.num_trial >= 5:
                    print('********************trial exceed********************')
                if PUT_BIN:
                    self.put_bin()
                else:
                    self.retract()
                # reset
                self.acc_points = np.zeros([4, 0])
                self.remaining_step = cfg.RL_MAX_STEP
                self.has_plan = False
                self.num_trial = 0
                return

            # run ga-ddpg
            print('use ga-ddpg')
            target_state = select_target_point(state) # only target points
            action, _, _, aux_pred = self.agent.select_action(target_state, remain_timestep=self.remaining_step)
            print('finish network') 
            pose_delta = unpack_action(action)
            ef_pose = get_ef_pose(self.pose_listener)
            ef_pose = ef_pose.dot(pose_delta)
            RT_grasp = ef_pose.dot(self.grasp_offset)
            vis_pose = ef_pose.copy()
            # send_transform(RT_grasp, vis_pose, 'GADDPG_action')
            self.moveit.go_local(RT_grasp, wait=True)
            print('remaining step: {} aggr. point: {}'.format(self.remaining_step, self.acc_points.shape[1]))
            # raw_input('next step?')
        
            self.remaining_step = max(self.remaining_step-1, 1)
            if self.remaining_step == 1:
                self.remaining_step += 5
                self.num_trial += 1


    def retract(self):
        """
        close finger and lift
        """    
        # close the gripper
        self.moveit.close_gripper(force=60)
        rospy.sleep(1.0)

        # lift object
        delta = 0.20
        joints = get_joints(self.joint_listener)
        T = self.moveit.forward_kinematics(joints[:-2])
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)
        # wait a few seconds
        rospy.sleep(2.0)

        # put object down
        T_put = T.copy()
        T_put[2, 3] += 0.01
        self.moveit.go_local(T_put, wait=True)
        self.moveit.open_gripper()
        self.moveit.go_local(T_lift, wait=True)

        if GA_DDPG_ONLY:
            self.moveit.retract()
        else:
            step = 20
            joint_position = get_joints(self.joint_listener)
            end_conf = np.append(self.moveit.home_q, joint_position[7:]) 
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]
            self.moveit.execute(traj)

        raw_input('finished. Try again?')


    # grasp object and put object into a bin with goal conf
    def put_bin(self):

        force_before = self.joint_listener.robot_force
        print('force before grasping', force_before)

        # close the gripper
        self.moveit.close_gripper(force=60)
        rospy.sleep(0.5)

        # lift object a bit
        delta = 0.05
        joints = get_joints(self.joint_listener)
        T = self.moveit.forward_kinematics(joints[:-2])
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)

        force_after = self.joint_listener.robot_force
        print('force after grasping', force_after)
        force_diff = np.linalg.norm(force_before - force_after)
        print('force diff norm', force_diff)

        # lift object more
        delta = 0.30
        joints = get_joints(self.joint_listener)
        T = self.moveit.forward_kinematics(joints[:-2])
        print('T in retract', T)
        T_lift = T.copy()
        T_lift[2, 3] += delta
        self.moveit.go_local(T_lift, wait=True)

        # check grasp success
        joint_position = self.joint_listener.joint_position
        print('check success', joint_position)
        if joint_position[-1] > 0.002 or force_diff > 0.5 or force_diff == 0:
            success = True
            print('grasp success')
        else:
            success = False
            print('grasp fail')

        # plan to goal conf
        step = 20
        if success:
            joint_position = get_joints(self.joint_listener)
            end_conf = np.append(self.bin_conf_1, joint_position[7:]) 
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]
            self.moveit.execute(traj)

            joint_position = get_joints(self.joint_listener)
            end_conf = np.append(self.bin_conf_2, joint_position[7:]) 
            traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]
            self.moveit.execute(traj)
            self.moveit.open_gripper()

        joint_position = get_joints(self.joint_listener)
        end_conf = np.append(self.moveit.home_q, joint_position[7:]) 
        traj = self.planner.plan_to_conf(step, joint_position, end_conf, vis=False)[::2, :]
        self.moveit.execute(traj)
        self.moveit.open_gripper()


    def bias_target_pc_regularize(self, point_state, total_point_num, target_pt_num=1024, use_farthest_point=True):
        target_mask = point_state[3, :] == 0        
        target_pt = point_state[:, target_mask]
        nontarget_pt = point_state[:, ~target_mask]
        print(target_pt.shape, nontarget_pt.shape)
        if target_pt.shape[1] > 0:
            target_pt = regularize_pc_point_count(target_pt.T, target_pt_num, use_farthest_point).T
        if nontarget_pt.shape[1] > 0:
            effective_target_pt_num = min(target_pt_num, target_pt.shape[1])
            nontarget_pt = regularize_pc_point_count(nontarget_pt.T, total_point_num - effective_target_pt_num, use_farthest_point).T
        return np.concatenate((target_pt, nontarget_pt), axis=1)


    # new_points is in hand coordinate
    # ACC_POINTS is in base
    def update_curr_acc_points(self, new_points, ef_pose, step):
        """
        Update accumulated points in world coordinate
        """
        new_points = se3_transform_pc(ef_pose, new_points)  
        # the number below can be adjusted for efficiency and robustness
        aggr_sample_point_num = min(int(CONFIG.pt_accumulate_ratio**step * CONFIG.uniform_num_pts), new_points.shape[1])
        index = np.random.choice(range(new_points.shape[1]), size=aggr_sample_point_num, replace=False).astype(np.int)

        new_points = new_points[:,index]
        print('new points before filtering with table height', new_points.shape)
        index = new_points[2, :] > self.table_height
        new_points = new_points[:, index]
        print('new points {} total point {}'.format(new_points.shape, self.acc_points.shape))

        self.acc_points = np.concatenate((new_points, self.acc_points), axis=1) #
        self.acc_points = regularize_pc_point_count(self.acc_points.T, 4096, use_farthest_point=True).T
        # if it still grows too much, can limit points by call regularize pc point count
        # self.planner.expert_plan can also be called with these dense points directly


    def goal_closure(self, action, goal):
        action_2 = np.zeros(7)
        action_2[-3:] = action[:3]
        action_2[:-3] = mat2quat(euler2mat(action[3], action[4], action[5])) # euler to quat

        point_dist = float(agent.goal_pred_loss(torch.from_numpy(goal)[None].float().cuda(), 
                            torch.from_numpy(action_2)[None].float().cuda()))
        print('point dist: {:.3f}'.format(point_dist))
        return point_dist < 0.008        


    def graspnet_closure(self, point_state):
        """
        Compute grasp quality from tf grasp net. 
        """
        score  = self.graspnet.compute_grasps_score(sess,  point_state)
        print('grasp closure score:', score)
        return score > self.grasp_score_threshold # tuned threshold


    # point_state is in hand coordinate
    def process_pointcloud(self, point_state, im_ef_pose, step, agg=True, use_farthest_point=False):
        """
        Process the cluttered scene point_state
        [0 - 6]: random or gripper points with mask -1
        [6 - 1030]: target point with mask 0
        [1030 - 5002]: obstacle point with mask 1
        [5002 - 5502]: robot points with mask 2 can be random or generated with get_collision_points and transform with joint
        """

        # accumulate all point state in base
        # set the mask 0 as target, 1 as other objects
        index_target = point_state[3, :] == self.target_obj_id
        index_other = point_state[3, :] != self.target_obj_id
        point_state[3, index_target] = 0.
        point_state[3, index_other] = 1.
 
        if agg: 
            self.update_curr_acc_points(point_state, im_ef_pose, step)

        # base to hand
        inv_ef_pose = se3_inverse(im_ef_pose)
        point_state = se3_transform_pc(inv_ef_pose, self.acc_points)
        point_state = self.bias_target_pc_regularize(point_state, CONFIG.uniform_num_pts)

        hand_finger_point = np.concatenate([self.hand_finger_point, np.ones((1, self.hand_finger_point.shape[1]), dtype=np.float32)], axis=0)
        point_state = np.concatenate([hand_finger_point, point_state], axis=1)
        point_state_ = point_state.copy()
        point_state_[3, :hand_finger_point.shape[1]] = -1
        # ignore robot points make sure it's 6 + 4096 + 500
        point_state_ = np.concatenate((point_state_, np.zeros((4, 500))), axis=1)
        point_state_[3, -500:] = 2
        return point_state_


    def camera_image_to_state(self, rgb, depth, mask, mask_background, im_ef_pose, step, agg=True, vis=False):
        """
        map from camera images and segmentations to object point cloud in robot coordinate
        mask: 0 represents target, 1 everything else
        mask: w x h x 1
        rgb:  w x h x 3
        depth:w x h x 1
        """
        if vis:     
            fig = plt.figure(figsize=(14.4, 4.8))
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow(rgb[:, :, (2, 1, 0)])
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(depth[...,0])
            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(mask)
            plt.show()

        mask_target = np.zeros_like(mask)
        mask_target[mask == self.target_obj_id] = 1
        mask_state = 1 - mask_target[...,None]
        image_state = np.concatenate([rgb, depth, mask_state], axis=-1)
        image_state = image_state.T
        
        # depth to camera, all the points on foreground objects
        # backproject depth
        depth_cuda = torch.from_numpy(depth).cuda()
        fx = INTRINSICS[0, 0]
        fy = INTRINSICS[1, 1]
        px = INTRINSICS[0, 2]
        py = INTRINSICS[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth_cuda)[0].cpu().numpy()

        # select points
        valid = (depth[...,0] != 0) * (mask > 0)
        point_xyz = im_pcloud[valid, :].reshape(-1, 3)
        label = mask[valid][...,None]
        point_state = np.concatenate((point_xyz, label), axis=1).T
        # point_state = backproject_camera_target_realworld_clutter(depth, INTRINSICS, mask)
        print('%d foreground points' % point_state.shape[1])
  
        # filter depth
        index = point_state[2, :] < self.depth_threshold
        point_state = point_state[:, index]

        # camera to hand
        point_state = se3_transform_pc(EXTRINSICS, point_state)

        # background points
        valid = (depth[...,0] != 0) * (mask_background > 0)
        point_background = im_pcloud[valid, :].reshape(-1, 3)
        index = point_background[:, 2] < self.depth_threshold
        point_background = point_background[index, :]
        if point_background.shape[0] > 0:
            point_background = regularize_pc_point_count(point_background, 1024, use_farthest_point=False)
            point_background = se3_transform_pc(EXTRINSICS, point_background.T)

        # accumate points in base, and transform to hand again
        point_state = self.process_pointcloud(point_state, im_ef_pose, step, agg)
        obs = (point_state, image_state)
        return obs, point_background


    # state points and grasp are in hand coordinate
    def vis_realworld(self, state, rgb, grasp, local_view=True, curr_joint=None):
        """
        visualize grasp and current observation 
        local view (hand camera view with projected points)
        global view (with robot and accumulated points) 
        this can be converted to ros
        """

        ef_pose = get_ef_pose(self.pose_listener)
        if local_view:
            print('in vis realworld local view')
            # base to hand
            points = se3_transform_pc(se3_inverse(ef_pose), self.acc_points)
            rgb = rgb[:,:,::-1]
            rgb = proj_point_img(rgb, INTRINSICS, se3_inverse(EXTRINSICS), points[:3], real_world=True)
            grasp = unpack_pose_rot_first(grasp) # .dot(rotZ(np.pi/2))
            rgb = draw_grasp_img(rgb, grasp, INTRINSICS, se3_inverse(EXTRINSICS), vis=True, real_world=True) 
            # show image
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(rgb)
            plt.show()
        else:
            print('in vis realworld global view')
            # global view
            point_color = [255, 255, 0]
            if curr_joint is None:
                curr_joint = get_joints(self.joint_listener)
                point_color = [0, 255, 0]
            poses_ = robot.forward_kinematics_parallel(
                                wrap_value(curr_joint)[None], offset=True)[0]
            grasp = poses_[7].dot(unpack_pose_rot_first(grasp)) 
            poses = [pack_pose(pose) for pose in poses_]
            line_starts, line_ends = grasp_gripper_lines(grasp[None])

            # green: observation, yellow: simulation, red: cage point
            cage_points_mask, depth_heuristics = self.compute_cage_point_mask( )
            noncage_points = self.acc_points[:3, ~cage_points_mask]
            cage_points = self.acc_points[:3, cage_points_mask]
            rgb = self.planner.planner_scene.renderer.vis(poses, list(range(10)), 
                shifted_pose=np.eye(4),
                interact=2,
                V=np.array(V),
                visualize_context={
                    "white_bg": True,
                    "project_point": [noncage_points, cage_points],
                    "project_color": [[0, 255, 0], [255, 0, 0]],
                    "static_buffer": True,
                    "reset_line_point": True,
                    "thickness": [2],
                    "line": [(line_starts[0], line_ends[0])],
                    "line_color": [[255, 0, 0]],            
                }
            )
        return rgb


    def compute_cage_point_mask(self):
        # points in global cooridnate
        index = self.acc_points[3, :] == 0
        points = self.acc_points[:3, index]

        # base to hand
        ef_pose = get_ef_pose(self.pose_listener)
        inv_ef_pose = se3_inverse(ef_pose)
        point_state = se3_transform_pc(inv_ef_pose, points)  
        
        # 0.11
        cage_points_mask =  (point_state[2] > 0.06)  * (point_state[2] < 0.09) * \
                            (point_state[1] > -0.05) * (point_state[1] < 0.05) * \
                            (point_state[0] > -0.02) * (point_state[0] < 0.02) 
        terminate =  cage_points_mask.sum() > CAGE_POINT_THRESHOLD
        # maybe this is more robust (use_farthest_point)?
        cage_points_mask_reg = regularize_pc_point_count(cage_points_mask[:,None], 
                               CONFIG.uniform_num_pts, use_farthest_point=False)
        print('number of cage points %d' % cage_points_mask_reg.sum())
        terminate  = cage_points_mask_reg.sum() > CAGE_POINT_THRESHOLD
        return cage_points_mask, terminate


    def termination_heuristics(self, state):
        """
        Target depth heuristics for determining if grasp can be executed.
        The threshold is based on depth in the middle of the camera and the finger is near the bottom two sides
        """

        point_state = state[0][0]
        target_mask = get_target_mask(point_state)
        point_state = point_state[:3, target_mask].T
        depth_heuristics = self.graspnet_closure(point_state)             
        if (depth_heuristics):
            print('object inside gripper? start retracting...')
        return depth_heuristics


    def preview_trajectory(self, state, remain_timestep, vis=False):
        """
        use the current point cloud to simulate observation and action for a trajectory
        this can be used to check trajectory before execution
        """
        print('in preview trajectory')
        state_origin = copy.deepcopy(state)
        sim_state = [state[0][0].copy(), state[0][1]] 

        joints = get_joints(self.joint_listener)
        ef_pose = get_ef_pose(self.pose_listener)
        ef_pose_origin = ef_pose.copy()
        joint_plan = [joints]
        ef_pose_plan = [ef_pose]

        for episode_steps in range(remain_timestep):
            state[0] = sim_state
            gaddpg_input_state = select_target_point(state)
            step = min(max(remain_timestep - episode_steps, 1), 25)
            action, _, _, aux_pred = agent.select_action(gaddpg_input_state, remain_timestep=step)
            action_pose = unpack_action(action)
            ef_pose = ef_pose.dot(action_pose)
            joints = solve_ik(joints, pack_pose(ef_pose))
            joint_plan.append(joints)
            ef_pose_plan.append(ef_pose)
            sim_next_point_state = se3_transform_pc(se3_inverse(action_pose), sim_state[0]) 
            sim_state[0] = sim_next_point_state

        if vis:
            # vis entire traj. Might be useful
            poses_ = robot.forward_kinematics_parallel(
                                wrap_value(joint_plan[0])[None], offset=True)[0]
            poses = [pack_pose(pose) for pose in poses_]
            line_starts, line_ends = grasp_gripper_lines(np.array(ef_pose_plan))
            points = state_origin[0][0]
            points = se3_transform_pc(ef_pose_origin, points)
            point_color = get_point_color(points)
            rgb = self.planner.planner_scene.renderer.vis(poses, list(range(10)), 
                shifted_pose=np.eye(4),
                interact=2,
                V=np.array(V),
                visualize_context={
                    "white_bg": True,
                    "project_point": [points],
                    "project_color": [point_color],
                    "static_buffer": True,
                    "reset_line_point": True,
                    "thickness": [2],
                    "line": [(line_starts[0], line_ends[0])],
                    "line_color": [[255, 0, 0]],            
                }
            )

        num = len(joint_plan)
        traj = np.zeros((num, 9), dtype=np.float32)
        for i in range(num):
            traj[i, :] = joint_plan[i]
        return traj


# for debuging
def send_transform(T, ef_pose, name, base_frame='measured/base_link'):
    broadcaster = tf.TransformBroadcaster()
    marker_pub = rospy.Publisher(name, Marker, queue_size = 10)
    for i in range(100):
        print('sending transformation {}'.format(name))
        qt = mat2quat(T[:3, :3])
        broadcaster.sendTransform(T[:3, 3], [qt[1], qt[2], qt[3], qt[0]], rospy.Time.now(), name, base_frame)

        GRASP_FRAME_OFFSET = tra.quaternion_matrix([0, 0, -0.707, 0.707])
        GRASP_FRAME_OFFSET[:3, 3] = [0, 0, 0.0]
        vis_pose = np.matmul(ef_pose, GRASP_FRAME_OFFSET)

        publish_grasps(marker_pub, base_frame, vis_pose)
        rospy.sleep(0.1)


def show_grasps(ef_poses, name, base_frame='measured/base_link'):
    marker_pub = rospy.Publisher(name, MarkerArray, queue_size = 10)
    GRASP_FRAME_OFFSET = tra.quaternion_matrix([0, 0, -0.707, 0.707])
    GRASP_FRAME_OFFSET[:3, 3] = [0, 0, 0.0]
    color = [0, 1, 0, 1]

    while not rospy.is_shutdown():
        markerArray = MarkerArray()
        for i in range(ef_poses.shape[0]):
            ef_pose = ef_poses[i]
            vis_pose = np.matmul(ef_pose, GRASP_FRAME_OFFSET)

            marker = create_gripper_marker_message (
                frame_id = base_frame,
                namespace = 'hand',
                mesh_resource = 'package://grasping_vae/panda_gripper.obj',
                color = color,
                marker_id = i,
            )
            pos = tra.translation_from_matrix(vis_pose)
            quat = tra.quaternion_from_matrix(vis_pose)
            marker.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
            markerArray.markers.append(marker)

        # Renumber the marker IDs
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        marker_pub.publish(markerArray)
        print('publishing grasps')
        rospy.sleep(0.1)


def create_gripper_marker_message(
        frame_id, 
        namespace,
        mesh_resource, 
        color, 
        lifetime=True, 
        mesh_use_embedded_materials=True,                 
        marker_id=0, 
        frame_locked=False,):
    marker = Marker()
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.ns = namespace
    if lifetime:
        marker.lifetime = rospy.Duration(0.2)
    marker.frame_locked = frame_locked
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.scale.x = marker.scale.y = marker.scale.z = 0.5
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = mesh_resource                                        
    marker.mesh_use_embedded_materials = mesh_use_embedded_materials            

    return marker


def publish_grasps(publisher, frame_id, grasp):
    color = [0, 1, 0, 1]
    marker = create_gripper_marker_message (
        frame_id=frame_id,
        namespace='hand',
        mesh_resource='package://grasping_vae/panda_gripper.obj',
        color=color,
        marker_id=0,
    )
    pos = tra.translation_from_matrix(grasp)
    quat = tra.quaternion_from_matrix(grasp)
    marker.pose = Pose(position=Point(*pos), orientation=Quaternion(*quat))
    publisher.publish(marker)



def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    """
    trans, rot = tf_pose
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


def gaddpg_grasps_from_simulate_view(gaddpg, state, time, ef_pose):
    """
    simulate views for gaddpg
    """
    n = 30
    mask = get_target_mask(state[0][0])
    point_state = state[0][0][:, mask]

    # hand to base
    point_state = se3_transform_pc(ef_pose, point_state)
    print('target point shape', point_state.shape)
    # target center is in base coordinate now
    target_center = point_state.mean(1)[:3]
    print('target center', target_center)

    # set up gaddpg
    img_state = state[0][1]
    gaddpg.policy.eval()
    gaddpg.state_feature_extractor.eval()

    # sample view (simulated hand) in base
    view_poses = np.array(sample_ef_view_transform(n, 0.2, 0.5, target_center, linspace=True, anchor=True))

    # base to view (simulated hand)
    inv_view_poses = se3_inverse_batch(view_poses)
    transform_view_points = np.matmul(inv_view_poses[:,:3,:3], point_state[:3]) + inv_view_poses[:,:3,[3]]
 
    # gaddpg generate grasps
    point_state_batch = torch.from_numpy(transform_view_points).cuda().float()
    time = torch.ones(len(point_state_batch) ).float().cuda() * 10. # time
    point_state_batch = torch.cat((point_state_batch, torch.zeros_like(point_state_batch)[:, [0]]), dim=1)
    policy_feat  = gaddpg.extract_feature(img_state, point_state_batch, value=False, time_batch=time) 
    _,_,_,gaddpg_aux  = gaddpg.policy.sample(policy_feat)
 
    # compose with ef poses   
    gaddpg_aux = gaddpg_aux.detach().cpu().numpy()
    unpacked_poses = [unpack_pose_rot_first(pose) for pose in gaddpg_aux]
    goal_pose_ws = np.matmul(view_poses, np.array(unpacked_poses)) # grasp to ef
    planner_cfg.external_grasps = goal_pose_ws

    # show_grasps(view_poses, 'grasps')
    # planner_cfg.external_grasps = view_poses
    # planner_cfg.external_grasps = np.concatenate((goal_pose_ws, view_poses), axis=0) # also visualize view
    planner_cfg.use_external_grasp = True


def select_target_point(state, target_pt_num=1024):
    """
    get target point cloud for gaddpg input
    """
    point_state = state[0][0]
    target_mask = get_target_mask(point_state)
    # removing gripper point later
    point_state = point_state[:4, target_mask] #  
    gripper_pc  = point_state[:4, :6] #  
    point_num  = min(point_state.shape[1], target_pt_num)
    obj_pc = regularize_pc_point_count(point_state.T, point_num, False).T
    point_state = np.concatenate((gripper_pc, obj_pc), axis=1)
    return [(point_state, state[0][1])] + state[1:]


def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """
    load_from_pretrain = args.pretrained is not None and os.path.exists(args.pretrained)

    if load_from_pretrain and not args.finetune:
        cfg_folder = args.pretrained
        cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)
        cfg.RL_MODEL_SPEC = os.path.join(cfg_folder, cfg.RL_MODEL_SPEC.split("/")[-1])
        dt_string = args.pretrained.split("/")[-1]
 
    else:
        if args.fix_output_time is None:
            dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        else:
            dt_string = args.fix_output_time

    model_output_dir = os.path.join(cfg.OUTPUT_DIR, dt_string)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    new_output_dir = not os.path.exists(model_output_dir) and not args.test
    print("Using config:")
    pprint.pprint(cfg)
    net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    return net_dict, dt_string


def solve_ik(joints, pose):
    """
    For simulating trajectory
    """
    ik =  robot.inverse_kinematics(pose[:3], ros_quat(pose[3:]), seed=joints[:7])
    if ik is not None:
        joints = np.append(np.array(ik), [0.04, 0.04])
    return joints


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--env-name', default="PandaYCBEnv")
    parser.add_argument('--policy', default="DDPG" )
    parser.add_argument('--seed', type=int, default=123456, metavar='N' )
      
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained', type=str, default=None, help='test one model')
    parser.add_argument('--test', action="store_true", help='test one model')
    parser.add_argument('--log', action="store_true", help='log')
    parser.add_argument('--render', action="store_true", help='rendering')
    parser.add_argument('--record', action="store_true", help='record video')
    parser.add_argument('--test_episode_num', type=int, default=10, help='number of episodes to test')
    parser.add_argument('--finetune', action="store_true", help='deprecated')
    parser.add_argument('--expert', action="store_true", help='generate experte rollout')
    parser.add_argument('--num_runs',  type=int, default=1)
    parser.add_argument('--max_cnt_per_obj',  type=int, default=10)
    parser.add_argument('--model_surfix',  type=str, default='latest', help='surfix for loaded model')
    parser.add_argument('--rand_objs', action="store_true", help='random objects in Shapenet')
    parser.add_argument('--load_test_scene', action="store_true", help='load pregenerated random scenes')
    parser.add_argument('--change_dynamics', action="store_true", help='change dynamics of the object')
    parser.add_argument('--egl', action="store_true", help='use egl plugin in bullet')

    parser.add_argument('--config_file',  type=str, default=None)
    parser.add_argument('--output_file',  type=str, default='rollout_success.txt')
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--use_external_grasp', action="store_true")
    parser.add_argument('--vis_grasp_net', action="store_true")
    parser.add_argument('--start_idx',  type=int, default=1)
    parser.add_argument('--real_world', action="store_true")
    parser.add_argument('--preview_traj', action="store_true")
    parser.add_argument('--fix_initial_state', action="store_true")

    args = parser.parse_args()
    return args, parser


### TODO
def get_joints(joint_listener):
    """
    (9, ) robot joint in radians 
    just for rendering and simulating
    """      
    if LOCAL_TEST: # dummy
        return np.array([-0.5596, 0.5123, 0.5575, -1.6929, 0.2937, 1.6097, -1.237, 0.04, 0.04])
    else:
        joints = joint_listener.joint_position
        print('robot joints', joints)
        return joints

  
def get_ef_pose(pose_listener):
    """
    (4, 4) end effector pose matrix from base 
    """  
    if LOCAL_TEST: # dummy
        return np.array([[-0.1915,  0.8724, -0.4498,  0.6041],
                         [ 0.7355,  0.4309,  0.5228, -0.0031],
                         [ 0.6499, -0.2307, -0.7242,  0.3213],
                         [ 0.,      0.,      0.,      1.    ]])
    else:
        base_frame = 'measured/base_link'
        target_frame = 'measured/panda_hand'
        try:
            tf_pose = pose_listener.lookupTransform(base_frame, target_frame, rospy.Time(0))
            pose = make_pose(tf_pose)
        except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
            pose = None
            print('cannot find end-effector pose')
            sys.exit(1)
        return pose


initial_joints = np.array([[-0.02535421982428639, -1.1120411124179306, 0.07915425984753728, -2.574433677700231, 0.0012470895074533914, 1.926161096378418, 0.9002216220876491],
                           [0.5805350207739269, -0.8111362388758844, -1.1146667134109263, -2.2735199081247064, -0.18589086490010281, 2.2351670468606946, -0.36534494081830765],
                           [-0.4345369377943954, -1.05069044781103, 1.119439285721959, -2.421638742837782, -0.02910207191286081, 2.0685257700621205, 1.5517931027048162],
                           [0.6299110230284048, -1.2067977417344766, -1.3116628687477672, -2.0905629379711166, -0.32998541843294193, 1.8464060782205653, -0.45038227560404887],
                           [-0.7665819353096028, -1.0393133004705655, 1.322218198802843, -2.0935060303990145, 0.33048455105753755, 1.8427947370070838, 1.746254150224718]])


if __name__ == '__main__':
    # Lirui: Replacing setup code
    # take a look at test_realworld for execution in ycb if necessary

    args, parser = parse_args()

    print('Called with args:')
    print(args)

    # create robot 
    rospy.init_node("gaddpg")

    from OMG.ycb_render.robotPose import robot_pykdl
    robot = robot_pykdl.robot_kinematics(None, data_path='../../../')
 
    ############################# DEFINE RENDERER
    '''
    from OMG.ycb_render.ycb_renderer import YCBRenderer
    width, height = 640, 480
    renderer = YCBRenderer(width=width, height=height, offset=False)
    renderer.set_projection_matrix(width, height, width * 0.8, width * 0.8, width / 2, height / 2, 0.1, 6)
    renderer.set_camera_default()

    models = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "finger", "finger"]
    obj_paths = ["data/robots/{}.DAE".format(item) for item in models]
    renderer.load_objects(obj_paths)
    '''

    V =     [
                [-0.9351, 0.3518, 0.0428, 0.3037],
                [0.2065, 0.639, -0.741, 0.132],
                [-0.2881, -0.684, -0.6702, 1.8803],
                [0.0, 0.0, 0.0, 1.0],
            ]
    CAGE_POINT_THRESHOLD = 25

    ############################# SETUP MODEL
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    net_dict, output_time = setup()
    CONFIG = cfg.RL_TRAIN
    LOCAL_TEST = False  # not using actual robot

    # Args
    ## updated
    if GA_DDPG_ONLY:
        cfg.RL_MAX_STEP = 20
    else:
        cfg.RL_MAX_STEP = 50
        CONFIG.uniform_num_pts = 4096

    CONFIG.output_time = output_time
    CONFIG.off_policy = True   
    POLICY = 'DDPG' if CONFIG.RL else 'BC'    	
    CONFIG.index_file = 'ycb_large.json'

    # The default config?
    cfg.ROS_CAMERA = 'D415'
    cfg.SCALES_BASE = [1.0]

    # Metrics
    input_dim = CONFIG.feature_input_dim
    cnt = 0.
    object_performance = {}
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)
    pretrained_path = model_output_dir

    # graspnet
    graspnet_cfg = get_graspnet_config(parser)
    graspnet_cfg = joint_config(
        graspnet_cfg.vae_checkpoint_folder,
        graspnet_cfg.evaluator_checkpoint_folder,
    )        
    graspnet_cfg['threshold'] = 0.8
    graspnet_cfg['sample_based_improvement'] = False
    graspnet_cfg['num_refine_steps'] = 5  # 20 
    graspnet_cfg['num_samples'] = 200       

    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    g1 = tensorflow.compat.v1.Graph()
    with g1.as_default():
        sess = tensorflow.compat.v1.Session(config=config)
        with sess.as_default():
            grasp_estimator = GraspEstimator(graspnet_cfg)
            grasp_estimator.build_network()
            grasp_estimator.load_weights(sess)

    if CONTACT_GRASPNET:
        graspnet_cfg_contact = get_graspnet_config_contact()
        global_config = config_utils.load_config(graspnet_cfg_contact.ckpt_dir, batch_size=graspnet_cfg_contact.forward_passes, arg_configs=graspnet_cfg_contact.arg_configs)

        # Create a session
        g2 = tensorflow.compat.v1.Graph()
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with g2.as_default():
            sess_contact = tensorflow.compat.v1.Session(config=config)
            with sess_contact.as_default():
                grasp_estimator_contact = GraspEstimatorContact(global_config)
                grasp_estimator_contact.build_network()
                saver = tensorflow.compat.v1.train.Saver(save_relative_paths=True)
                grasp_estimator_contact.load_weights(sess_contact, saver, graspnet_cfg_contact.ckpt_dir, mode='test')
    else:
        grasp_estimator_contact = None

    # GA-DDPG
    action_space = PandaTaskSpace6D()  
    agent = globals()[POLICY](input_dim, action_space, CONFIG) # 138  
    agent.setup_feature_extractor(net_dict, args.test)
    agent.load_model(pretrained_path, surfix=args.model_surfix, set_init_step=True)

    ############################# DEFINE ROS INTERFACE 
    listener = ImageListener(agent, grasp_estimator, grasp_estimator_contact)
    while not rospy.is_shutdown():
       listener.run_network()
