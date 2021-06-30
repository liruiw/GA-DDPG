# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import random

from scipy import interpolate
import scipy.io as sio
import IPython
import time
from torch import nn
from torch import optim

import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
import tabulate
import yaml
import torch

from core import networks
import copy
import math
from easydict import EasyDict as edict
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from torch.optim import Adam 
from collections import deque
import psutil
import GPUtil

hand_finger_point = np.array([ [ 0.,  0.,  0.   , -0.   ,  0.   , -0.   ],
                               [ 0.,  0.,  0.053, -0.053,  0.053, -0.053],
                               [ 0.,  0.,  0.075,  0.075,  0.105,  0.105]])
anchor_seeds = np.array([
                        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785],
                        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85],
                        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27],
                        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74],
                        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56],
                        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56],
                        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35],
                        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24],
                        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23]
                        ])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sum_2 = 0
        self.count_2 = 0
        self.means = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val * n
        self.count_2 += n

    def set_mean(self):
        self.means.append(self.sum_2 / self.count_2)
        self.sum_2 = 0
        self.count_2 = 0

    def std(self):
        return np.std(np.array(self.means) + 1e-4)

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)

def module_max_param(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data))
                for name, param in module.named_parameters()])
    return max_data


def module_max_gradient(module):
    def maybe_max(x):
        return torch.abs(x).max().item() if x is not None else 0

    max_grad = np.amax(
        [(maybe_max(param.grad)) for name, param in module.named_parameters()]
    )
    return max_grad
 
def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    R = np.stack([side, up, -forward], axis=-1)
    return R

def rand_sample_joint(env, init_joints=None, near=0.2, far=0.5):
    """
    randomize initial joint configuration
    """
    init_joints_ = env.randomize_arm_init(near, far)
    init_joints = init_joints_ if init_joints_ is not None else init_joints
    return init_joints

def check_scene(env, state, start_rot, object_performance=None, scene_name=None, 
                        init_dist_low=0.2, init_dist_high=0.5, run_iter=0):
    """
    check if a scene is valid by its distance, view, hand direction, target object state, and object counts
    """
    MAX_TEST_PER_OBJ = 10
    dist = np.linalg.norm( env._get_target_relative_pose('tcp')[:3, 3])
    dist_flag = dist > init_dist_low  and dist < init_dist_high
    pt_flag = state[0][0].shape[1] > 100
    z = start_rot[:3, 0] / np.linalg.norm(start_rot[:3,0])
    hand_dir_flag = z[-1] > -0.3
    target_obj_flag = env.target_name != 'noexists'
    if object_performance is None:
        full_flag = True
    else:
        full_flag = env.target_name not in object_performance or object_performance[env.target_name][0].count < (run_iter + 1) * MAX_TEST_PER_OBJ   
    name_flag  = 'pitcher' not in env.target_name
    return full_flag and target_obj_flag and pt_flag and name_flag  
 

def merge_two_dicts(x, y):
    z = x.copy()  
    z.update(y)    
    return z

def process_image_input(state):
    state[:, :3] *= 255
    if state.shape[1] >= 4:
        state[:, 3] *= 5000
    if state.shape[1] == 5:
        state[:, -1][state[:, -1] == -1] = 50
    return state.astype(np.uint16)

def check_ngc():
    GPUs = GPUtil.getGPUs()
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    return (gpu_limit > 14000)
    
def process_image_output(sample):
    sample = sample.astype(np.float32).copy()
    n = len(sample)
    if len(sample.shape) <= 2:
        return sample

    sample[:, :3] /= 255.0
    if sample.shape[0] >= 4:
        sample[:, 3] /= 5000
    sample[:, -1] = sample[:, -1] != 0
    return sample


def make_nets_opts_schedulers(model_spec, config, cuda_device="cuda"):
    specs = yaml.load(open(model_spec).read(), Loader=yaml.SafeLoader)  #
    ret = {}
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    for net_name, spec in specs.items():
        net_args = spec.get("net_kwargs", {})
        net_args["input_dim"] = config.channel_num

        if net_name == "state_feature_extractor":
            if hasattr(config, "policy_extra_latent"):
                net_args["policy_extra_latent"] = config.policy_extra_latent
                net_args["critic_extra_latent"] = config.critic_extra_latent
            if config.sa_channel_concat:
                spec["net_kwargs"]["action_concat"] = True

        net_class = getattr(networks, spec["class"])
        net = net_class(**net_args)
        net = torch.nn.DataParallel(net)
        if cuda_device is not None:
            net = net.to(cuda_device)

        d = {
            "net": net,
        }

        if "opt" in spec:
            d["opt"] = getattr(optim, spec["opt"])(
                net.parameters(), **spec["opt_kwargs"]
            )
            if len(config.overwrite_feat_milestone) > 0:
                spec["scheduler_kwargs"]["milestones"] = config.overwrite_feat_milestone
            print("schedule:", spec["scheduler_kwargs"]["milestones"])

            d["scheduler"] = getattr(optim.lr_scheduler, spec["scheduler"])(
                d["opt"], **spec["scheduler_kwargs"]
            )
            if hasattr(net.module, "encoder"):
                d["encoder_opt"] = getattr(optim, spec["opt"])(
                    net.module.encoder.parameters(), **spec["opt_kwargs"]
                )
                d["encoder_scheduler"] = getattr(optim.lr_scheduler, spec["scheduler"])(
                    d["encoder_opt"], **spec["scheduler_kwargs"]
                )
            if hasattr(net.module, "value_encoder"):
                d["val_encoder_opt"] = getattr(optim, spec["opt"])(
                    net.module.value_encoder.parameters(), **spec["opt_kwargs"]
                )
                d["val_encoder_scheduler"] = getattr(
                    optim.lr_scheduler, spec["scheduler"]
                )(d["val_encoder_opt"], **spec["scheduler_kwargs"])

        ret[net_name] = d
    return ret


def get_valid_index(arr, index):
    return arr[min(len(arr) - 1, index)]


def fc(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes), nn.LeakyReLU(0.1, inplace=True)
        )

def deg2rad(deg):
    if type(deg) is list:
        return [x/180.0*np.pi for x in deg]
    return deg/180.0*np.pi

def rad2deg(rad):
    if type(rad) is list:
        return [x/np.pi*180 for x in rad]
    return rad/np.pi*180

def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPG
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def projection_to_intrinsics(mat, width=224, height=224):
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def view_to_extrinsics(mat):
    pose = np.linalg.inv(np.array(mat).reshape([4, 4]).T)
    return np.linalg.inv(pose.dot(rotX(np.pi)))   


def concat_state_action_channelwise(state, action):
    """
    concate the action in the channel space
    """
    action = action.unsqueeze(2)
    state = torch.cat((state, action.expand(-1, -1, state.shape[2])), 1)
    return state

def safemat2quat(mat):
    quat = np.array([1,0,0,0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat

def se3_transform_pc(pose, point):
    if point.shape[1] == 3:
        return pose[:3, :3].dot(point) + pose[:3, [3]]
    else:
        point_ = point.copy()
        point_[:3] = pose[:3, :3].dot(point[:3]) + pose[:3, [3]]
        return point_

def has_check(x, prop):
    return hasattr(x, prop) and getattr(x, prop)

def migrate_model(in_model, out_model, surfix="latest", grasp_model=None):
    in_policy_name, out_policy_name = "BC", "DDPG"
    for file in [
        "actor_PandaYCBEnv_{}".format(surfix),
        "state_feat_PandaYCBEnv_{}".format(surfix),
        "goal_feat_PandaYCBEnv_{}".format(surfix),
        "critic_PandaYCBEnv_{}".format(surfix),
    ]:
        if not os.path.exists("{}/{}_{}".format(in_model, in_policy_name, file)):
            in_policy_name = "DDPG"
        cmd = "cp {}/{}_{} {}/{}_{}".format(
            in_model, in_policy_name, file, out_model, out_policy_name, file
        )
        if os.path.exists('{}/{}_{}'.format(in_model, in_policy_name, file)):
            os.system(cmd)
            print(cmd)
 

def depth_termination_heuristics(depth_img, mask_img):
    """
    depth_img, mask_img: w x h
    recommend scale to square 64 x 64
    """
    window_width, window_height = depth_img.shape
    depth_img = depth_img.copy()
    nontarget_mask = mask_img[...,0] != 0  
    min_depth = 0.105 # https://www.intelrealsense.com/depth-camera-d435/
    terminate_thre_depth = 0.045

    if use_depth_heuristics:        
        depth_img = depth_img[...,0] 
        depth_img[nontarget_mask] = 10  
        
        # hard coded region
        scale = window_width / 64
        depth_img_roi = depth_img[int(38. * scale) , int(24. * scale):int(48 * scale)] 
        depth_img_roi_ = depth_img_roi[depth_img_roi < 0.2]          
        if depth_img_roi_.shape[0] > 1:
            depth_heuristics = (depth_img_roi_ < terminate_thre_depth).sum() > 10 * scale 


def get_info(state, opt="img", IMG_SIZE=(112, 112)):
    if opt == "img":
        return (state[0][1][:3].T * 255).astype(np.uint8)
    if opt == "intr":
        cam_proj = np.array(state[-2][48:]).reshape([4, 4])
        return projection_to_intrinsics(cam_proj, IMG_SIZE[0], IMG_SIZE[1])[:3, :3]
    if opt == "point":
        return state[0][0][:3].T.copy()
 

def write_video(
    traj,
    scene_file,
    expert_traj=None,
    cnt=0,
    IMG_SIZE=(112, 112),
    output_dir="output_misc/",
    logdir="policy",
    target_name="",
    surfix="",
    use_pred_grasp=False,
    success=False,
    use_value=False,
):

    ratio = 1 if expert_traj is None else 2
    result = "success" if success else "failure"
    print('video path:', os.path.join(
            output_dir,
            "rl_output_video_{}/{}_rollout.avi".format(surfix,  int(cnt)),
        ))
    video_writer = make_video_writer(
        os.path.join(
            output_dir,
            "rl_output_video_{}/{}_rollout.avi".format(surfix,  int(cnt)),
        ),
        int(ratio * IMG_SIZE[0]),
        int(IMG_SIZE[1]),
    )
    text_color = [255, 0, 0] if use_pred_grasp else [0, 255, 0]
    for i in range(len(traj)):
        img = traj[i][..., [2, 1, 0]]
        if expert_traj is not None:
            idx = min(len(expert_traj) - 1, i)
            img = np.concatenate((img, expert_traj[idx][..., [2, 1, 0]]), axis=1)

        video_writer.write(img.astype(np.uint8))
        
def make_gripper_pts(points, color=(1, 0, 0)):
    # o3d.visualization.RenderOption.line_width = 8.0
    line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

    cur_gripper_pts = points.copy()
    cur_gripper_pts[1] = (cur_gripper_pts[2] + cur_gripper_pts[3]) / 2.0
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(cur_gripper_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_index)
    line_set.colors = o3d.utility.Vector3dVector(
        [color for i in range(len(line_index))]
    )
    return line_set
 
def _cross_matrix(x):
    """
    cross product matrix
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def a2e(q):
    p = np.array([0, 0, 1])
    r = _cross_matrix(np.cross(p, q))
    Rae = np.eye(3) + r + r.dot(r) / (1 + np.dot(p, q))
    return mat2euler(Rae)

def get_camera_constant(width):
    K = np.eye(3)
    K[0,0]=K[0,2]=K[1,1]=K[1,2] = width / 2.0
 
    offset_pose = np.zeros([4, 4])
    offset_pose[0,1]=-1.
    offset_pose[1,0]=offset_pose[2,2]=offset_pose[3,3]=1.
    offset_pose[2,3]=offset_pose[1,3]=-0.036
    return offset_pose, K

def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new
 
def backproject_camera_target(im_depth, K, target_mask):  
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    mask = (depth != 0) * (target_mask.flatten() == 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())  #
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )   
    X[1] *= -1  # flip y OPENGL. might be required for real-world
    return X[:, mask]

def backproject_camera_target_realworld(im_depth, K, target_mask):   
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    mask = (depth != 0) * (target_mask.flatten() == 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())  #
    X = np.multiply(
        np.tile(depth.reshape(1, width * height), (3, 1)), R
    )
    return X[:, mask]

def proj_point_img(img, K, offset_pose, points, color=(255, 0, 0), vis=False, neg_y=True, real_world=False): 
    xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]
    if real_world:
        pass
    elif neg_y: xyz_points[:2] *= -1
    p_xyz = K.dot(xyz_points)
    p_xyz = p_xyz[:, p_xyz[2] > 0.03]
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1)
    img[y[valid_idx_mask], x[valid_idx_mask]] = (0,255,0)
    return img

class PandaTaskSpace6D():
    def __init__(self):
        self.high = np.array([0.06,   0.06,  0.06,  np.pi/6,  np.pi/6,  np.pi/6]) #, np.pi/10
        self.low  = np.array([-0.06, -0.06, -0.06, -np.pi/6, -np.pi/6, -np.pi/6]) # , -np.pi/3
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])


def get_hand_anchor_index_point():
    hand_anchor_points = np.array(
        [
            [0, 0, 0],
            [0.00, -0.00, 0.058],
            [0.00, -0.043, 0.058],
            [0.00, 0.043, 0.058],
            [0.00, -0.043, 0.098],
            [0.00, 0.043, 0.098],
        ]
    )
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]
    return hand_anchor_points, line_index

def grasp_gripper_lines(pose):
    hand_anchor_points, line_index = get_hand_anchor_index_point()
    hand_points = (
        np.matmul(pose[:, :3, :3], hand_anchor_points.T) + pose[:, :3, [3]]
    )
    hand_points = hand_points.transpose([1, 0, 2])
    p1 = hand_points[:, :, line_index[0]].reshape([3, -1])
    p2 = hand_points[:, :, line_index[1]].reshape([3, -1])
    return [p1], [p2]

def draw_grasp_img(img, pose, K, offset_pose, color=(255, 0, 0), vis=False, real_world=False):
    img_cpy = img.copy()
    hand_anchor_points = np.array(
        [
            [0, 0, 0],
            [0.00, -0.00, 0.068],
            [0.00, -0.043, 0.068],
            [0.00, 0.043, 0.068],
            [0.00, -0.043, 0.108],  # 09
            [0.00, 0.043, 0.108],
        ]
    )
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]

    hand_anchor_points = pose[:3, :3].dot(hand_anchor_points.T) + pose[:3, [3]]
    hand_anchor_points = (
        offset_pose[:3, :3].dot(hand_anchor_points) + offset_pose[:3, [3]]
    )
    if not real_world:
        hand_anchor_points[:2] *= -1
    p_xyz = K.dot(hand_anchor_points)
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    x = np.clip(x, 0, img.shape[0] - 1)
    y = np.clip(y, 0, img.shape[1] - 1)
    for i in range(len(line_index[0])):
        pt1 = (x[line_index[0][i]], y[line_index[0][i]])
        pt2 = (x[line_index[1][i]], y[line_index[1][i]])
        cv2.line(img_cpy, pt1, pt2, color, 1)

    return img_cpy

def get_noise_delta(action, noise_level, noise_type="uniform"):
    normal = noise_type != "uniform"

    if type(action) is not np.ndarray:
        if normal:
            noise_delta = torch.randn_like(action) * noise_level / 2.0
        else:
            noise_delta = (torch.rand_like(action) * 3 - 6) * noise_level
        noise_delta[:, 3:] *= 5

    else:
        if normal:
            noise_delta = np.random.normal(size=(6,)) * noise_level / 2.0
        else:
            noise_delta = np.random.uniform(-3, 3, size=(6,)) * noise_level
        noise_delta[3:] *= 5  # radians
    return noise_delta

def unpack_action(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return pose_delta

def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked

def quat2euler(quat):
    return mat2euler(quat2mat(quat))


def pack_pose(pose, rot_first=False):
    packed = np.zeros(7)
    if rot_first:
        packed[4:] = pose[:3, 3]
        packed[:4] = safemat2quat(pose[:3, :3])
    else:
        packed[:3] = pose[:3, 3]
        packed[3:] = safemat2quat(pose[:3, :3])
    return packed


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX

 

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
 
 
def unpack_pose_rot_first(pose):
    unpacked = np.eye(4)
    unpacked[:3, :3] = quat2mat(pose[:4])
    unpacked[:3, 3] = pose[4:]
    return unpacked
 
def pack_pose_rot_first(pose):
    packed = np.zeros(7)
    packed[4:] = pose[:3, 3]
    packed[:4] = safemat2quat(pose[:3, :3])
    return packed

def inv_pose(pose):
    return pack_pose(np.linalg.inv(unpack_pose(pose)))


def relative_pose(pose1, pose2):
    return pack_pose(np.linalg.inv(unpack_pose(pose1)).dot(unpack_pose(pose2)))


def compose_pose(pose1, pose2):
    return pack_pose(unpack_pose(pose1).dot(unpack_pose(pose2)))


def safe_div(dividend, divisor, eps=1e-8):  # mark
    return dividend / (divisor + eps)


def wrap_value(value):
    if value.shape[0] <= 7:
        return rad2deg(value)
    value_new = np.zeros(value.shape[0] + 1)
    value_new[:7] = rad2deg(value[:7])
    value_new[8:] = rad2deg(value[7:])
    return value_new
 

def skew_matrix(r):
    """
    Get skew matrix of vector.
    r: 3 x 1
    r_hat: 3 x 3
    """
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def inv_relative_pose(pose1, pose2, decompose=False):
    """
    pose1: b2a
    pose2: c2a
    relative_pose:  b2c
    shape: (7,)
    """

    from_pose = np.eye(4)
    from_pose[:3, :3] = quat2mat(pose1[3:])
    from_pose[:3, 3] = pose1[:3]
    to_pose = np.eye(4)
    to_pose[:3, :3] = quat2mat(pose2[3:])
    to_pose[:3, 3] = pose2[:3]
    relative_pose = se3_inverse(to_pose).dot(from_pose)
    return relative_pose

def get_usage():
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    return gpu_usage, memory_usage


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat


def soft_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def half_soft_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        if target_name[:7] in ["linear1", "linear2", "linear3"]:  # polyak for target 1
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def half_hard_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        if target_name[:7] in ["linear4", "linear5", "linear6"]:  # polyak for target 1
            target_param.data.copy_(param.data)

def hard_update(target, source, tau=None):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def distance_by_translation_point(p1, p2):
    """
    Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            pc = torch.from_numpy(pc).cuda()[None].float() 
            new_xyz = (
            gather_operation(
                pc.transpose(1,2).contiguous(), furthest_point_sample(pc[...,:3].contiguous(), npoints)
            )
            .contiguous()
            )                                                                             
            pc = new_xyz[0].T.detach().cpu().numpy() 
            
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False
            )
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def get_control_point_tensor(batch_size, use_torch=True, device="cpu", rotz=False):
    """
    Outputs a tensor of shape (batch_size x 6 x 3).
    use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.array([[ 0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ],
       [ 0.053, -0.   ,  0.075],
       [-0.053,  0.   ,  0.075],
       [ 0.053, -0.   ,  0.105],
       [-0.053,  0.   ,  0.105]], dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])
    if rotz:
        control_points = np.matmul(control_points, rotZ(np.pi / 2)[:3, :3])
    if use_torch:
        return torch.tensor(control_points).to(device).float()

    return control_points.astype(np.float32)


def transform_control_points(
    gt_grasps,
    batch_size,
    mode="qt",
    device="cpu",
    t_first=False,
    rotz=False,
    control_points=None,
):
    """
    Transforms canonical points using gt_grasps.
    mode = 'qt' expects gt_grasps to have (batch_size x 7) where each
      element is catenation of quaternion and translation for each
      grasps.
    mode = 'rt': expects to have shape (batch_size x 4 x 4) where
      each element is 4x4 transformation matrix of each grasp.
    """
    assert mode == "qt" or mode == "rt", mode
    grasp_shape = gt_grasps.shape
    if grasp_shape[-1] == 7:
        assert len(grasp_shape) == 2, grasp_shape
        assert grasp_shape[-1] == 7, grasp_shape
        if control_points is None:
            control_points = get_control_point_tensor(
                batch_size, device=device, rotz=rotz
            )
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps, 1).repeat(1, num_control_points, 1)

        if t_first:
            gt_q = gt_grasps[:, :, 3:]
            gt_t = gt_grasps[:, :, :3]
        else:
            gt_q = gt_grasps[:, :, :4]
            gt_t = gt_grasps[:, :, 4:]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert len(grasp_shape) == 3, grasp_shape
        assert grasp_shape[1] == 4 and grasp_shape[2] == 4, grasp_shape
        if control_points is None:
            control_points = get_control_point_tensor(
                batch_size, device=device, rotz=rotz
            )
        shape = control_points.shape
        ones = torch.ones(
            (shape[0], shape[1], 1), dtype=torch.float32, device=control_points.device
        )
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))
 

def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1.0, 0.0, 0.0], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))
 
def control_points_from_rot_and_trans(
    grasp_eulers, grasp_translations, device="cpu", grasp_pc=None
):
    rot = tc_rotation_matrix(
        grasp_eulers[:, 0], grasp_eulers[:, 1], grasp_eulers[:, 2], batched=True
    )
    if grasp_pc is None:
        grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], device=device)

    grasp_pc = torch.matmul(grasp_pc.float(), rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1], -1)
    return grasp_pc


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def get_policy_class(policy_net_name, args):
    policy = networks.GaussianPolicy(
        args.num_inputs,
        args.action_dim,
        args.hidden_size,
        args.action_space,
        extra_pred_dim=args.extra_pred_dim,
         
    ).to('cuda')
    policy_optim = Adam(
       policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-5 )
    policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        policy_optim, milestones=args.policy_milestones, gamma=args.lr_gamma)
    policy_target = getattr(networks, policy_net_name)(
        args.num_inputs,
        args.action_dim,
        args.hidden_size,
        args.action_space,
        extra_pred_dim=args.extra_pred_dim,
         
    ).to('cuda')
    return policy, policy_optim, policy_scheduler, policy_target


def get_critic(args):
    model = networks.QNetwork
    critic = model(
            args.critic_num_input,
            args.critic_value_dim,
            args.hidden_size,
            extra_pred_dim=args.critic_extra_pred_dim,
        ).cuda()

    critic_optim = Adam(
        critic.parameters(), lr=args.value_lr, eps=1e-5, weight_decay=1e-5 )
    critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        critic_optim,
        milestones=args.value_milestones,
        gamma=args.value_lr_gamma,
    )
    critic_target = model(
        args.critic_num_input,
        args.critic_value_dim,
        args.hidden_size,
        extra_pred_dim=args.critic_extra_pred_dim,
    ).cuda()
    return critic, critic_optim, critic_scheduler, critic_target

def get_loss_info_dict():
   return {     'bc_loss': deque([0], maxlen=50),
                'policy_grasp_aux_loss': deque([0], maxlen=50),
                'critic_grasp_aux_loss': deque([0], maxlen=100),
                'critic_loss': deque([0], maxlen=100),
                'actor_critic_loss': deque([0], maxlen=50), 
                'reward_mask_num': deque([0], maxlen=5),
                'expert_mask_num': deque([0], maxlen=5),
                'policy_param': deque([0], maxlen=5),
                'critic_grad': deque([0], maxlen=5),
                'critic_param': deque([0], maxlen=5),     
                'train_batch_size': deque([0], maxlen=5) 
             }    