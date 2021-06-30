# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
import time
import sys

import pybullet as p
import numpy as np
import IPython

from env.panda_gripper_hand_camera import Panda
from transforms3d.quaternions import *
import scipy.io as sio
from core.utils import *
import json
from itertools import product

BASE_LINK = -1
MAX_DISTANCE = 0.000
try:
    import OMG.ycb_render.robotPose.robot_pykdl as robot_pykdl
    from OMG.omg.config import cfg as planner_cfg
    from OMG.omg.core import PlanningScene
except:
    pass

def get_num_joints(body, CLIENT=None):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_links(body, CLIENT=None):
    return list(range(get_num_joints(body, CLIENT)))

def get_all_links(body, CLIENT=None):
    return [BASE_LINK] + list(get_links(body, CLIENT))

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE, CLIENT=None): # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)) != 0

def any_link_pair_collision(body1, body2, links1=None, links2=None, CLIENT=None, **kwargs):
    if links1 is None:
        links1 = get_all_links(body1, CLIENT)
    if links2 is None:
        links2 = get_all_links(body2, CLIENT)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, CLIENT=CLIENT, **kwargs):
            return True
    return False

def body_collision(body1, body2, max_distance=MAX_DISTANCE, CLIENT=None): # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, body2, links1, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

class PandaJointSpace():
    def __init__(self):
        self.high = np.ones(7) * 0.25
        self.low  = np.ones(7) * -0.25
        self.shape = [7]
        self.bounds = np.vstack([self.low, self.high])

class PandaTaskSpace6D():
    def __init__(self):
        self.high = np.array([0.06,   0.06,  0.06,  np.pi/6,  np.pi/6,  np.pi/6]) #, np.pi/10
        self.low  = np.array([-0.06, -0.06, -0.06, -np.pi/6, -np.pi/6, -np.pi/6]) # , -np.pi/3
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])

class PandaYCBEnv():
    """
    Class for franka panda environment with YCB objects.
    """

    def __init__(self,
                 renders=False,
                 maxSteps=100,
                 random_target=False,
                 blockRandom=0.5,
                 cameraRandom=0,
                 action_space='configuration',
                 use_expert_plan=False,
                 accumulate_points=False,
                 use_hand_finger_point=False,
                 expert_step=20,
                 expert_dynamic_timestep=False,
                 data_type='RGB',
                 filter_objects=[],
                 img_resize=(224, 224),
                 regularize_pc_point_count=False,
                 egl_render=False,
                 width=224,
                 height=224,
                 uniform_num_pts=1024,
                 numObjects=7,
                 termination_heuristics=True,
                 domain_randomization=False,
                 change_dynamics=False,
                 pt_accumulate_ratio=0.95,
                 initial_near=0.2,
                 initial_far=0.5,
                 disable_unnece_collision=True,
                 omg_config=None):

        self._timeStep = 1. / 1000.
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps
        self._env_step = 0
        self._resize_img_size = img_resize

        self._p = p
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._random_target = random_target
        self._accumulate_points = accumulate_points
        self._use_expert_plan = use_expert_plan
        self._expert_step = expert_step
        self._use_hand_finger_point = use_hand_finger_point
        self._data_type  = data_type
        self._egl_render = egl_render
        self._action_space = action_space
        self._disable_unnece_collision = disable_unnece_collision

        self._pt_accumulate_ratio = pt_accumulate_ratio
        self._change_dynamics = change_dynamics
        self._domain_randomization = domain_randomization
        self._initial_near = initial_near
        self._initial_far  = initial_far
        self._expert_dynamic_timestep = expert_dynamic_timestep
        self._termination_heuristics = termination_heuristics
        self._filter_objects = filter_objects
        self._omg_config = omg_config
        self._regularize_pc_point_count = regularize_pc_point_count
        self._uniform_num_pts = uniform_num_pts
        self.observation_dim = (self._window_width, self._window_height, 3)

        self.init_constant()
        self.connect()

    def init_constant(self):
        self._shift = [0.8, 0.8, 0.8] # to work without axis in DIRECT mode
        self._max_episode_steps = 50
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.data_root_dir = os.path.join(self.root_dir, 'data/scenes')
        self._planner_setup = False
        self.retracted = False
        self._standoff_dist = 0.08

        self.cam_offset = np.eye(4)
        self.cam_offset[:3, 3]  = (np.array([0.036, 0, 0.036]))   # camera offset
        self.cam_offset[:3, :3] = euler2mat(0,0,-np.pi/2)
        self.cur_goal = np.eye(4)

        self.target_idx = 0
        self.objects_loaded = False
        self.parallel = False
        self.curr_acc_points = np.zeros([3, 0])
        self.connected = False
        self.action_dim = 6
        self.hand_finger_points = hand_finger_point
        self.action_space =  PandaTaskSpace6D()

    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)

            p.resetDebugVisualizerCamera(1.3, 180.0, -41.0, [-0.35, -0.58, -0.88])

        else:
            self.cid = p.connect(p.DIRECT )

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True

    def disconnect(self):
        """
        Disconnect pybullet.
        """
        p.disconnect()
        self.connected = False

    def reset(self, save=False, init_joints=None, scene_file=None,
                        data_root_dir=None, cam_random=0,
                        reset_free=False, enforce_face_target=False ):
        """
        Environment reset called at the beginning of an episode.
        """
        self.retracted = False
        if data_root_dir is not None:
            self.data_root_dir = data_root_dir

        self._cur_scene_file = scene_file

        if reset_free:
            return self.cache_reset(scene_file, init_joints, enforce_face_target )

        self.disconnect()
        self.connect()

        # Set the camera  .
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0,0,-9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf') # _white
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, self.table_pos[0], self.table_pos[1], self.table_pos[2],
                             0.707, 0., 0., 0.707)

        # Intialize robot and objects
        if init_joints is None:
            self._panda = Panda(stepsize=self._timeStep, base_shift=self._shift)

        else:
            self._panda = Panda(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(1000):
                p.stepSimulation()

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()
            if self._use_expert_plan: self.setup_expert_scene()

        if  scene_file is None or not os.path.exists(os.path.join(self.data_root_dir, scene_file + '.mat')):
            self._randomly_place_objects(self._get_random_object(self._numObjects), scale=1)
        else:
            self.place_objects_from_scene(scene_file)

        self._objectUids += [self.plane_id, self.table_id]
        self._env_step = 0
        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        self.curr_acc_points = np.zeros([3, 0])
        return None  # observation

    def step(self, action, delta=False, obs=True, repeat=None, config=False, vis=False):
        """
        Environment step.
        """
        repeat = 150
        action = self.process_action(action, delta, config)
        self._panda.setTargetPositions(action)
        for _ in range(int(repeat)):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

        observation = self._get_observation(vis=vis)
        test_termination_obs =  observation[0][1]
        depth = test_termination_obs[[3]].T
        mask = test_termination_obs[[4]].T
        observation = self.input_selection(observation)

        done = self._termination(depth.copy(), mask, use_depth_heuristics=self._termination_heuristics)
        self.collision_check()
        reward = self._reward()

        info = { 'grasp_success': reward,
                 'goal_dist':self._get_goal_dist(),
                 'point_num':self.curr_acc_points.shape[1],
                 'collided': self.collided,
                 'cur_ef_pose':self._get_ef_pose(mat=True)}

        self._env_step += 1
        return observation, reward, done, info

    def _get_observation(self, pose=None, vis=False, acc=True ):
        """
        Get observation
        """

        object_pose = self._get_target_relative_pose('ef') # self._get_relative_ef_pose()
        ef_pose = self._get_ef_pose('mat')

        joint_pos, joint_vel = self._panda.getJointStates()
        near, far = self.near, self.far
        view_matrix, proj_matrix = self._view_matrix, self._proj_matrix
        extra_overhead_camera = False
        camera_info = tuple(view_matrix) + tuple(proj_matrix)
        hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, near, far = self._get_hand_camera_view(pose)
        camera_info += tuple(hand_cam_view_matrix.flatten()) + tuple(hand_proj_matrix)
        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                     height=self._window_height,
                                                     viewMatrix=tuple(hand_cam_view_matrix.flatten()),
                                                     projectionMatrix=hand_proj_matrix,
                                                     physicsClientId=self.cid,
                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16) # transform depth from NDC to actual depth
        mask[mask >= 0] += 1  # transform mask to have target id 0
        target_idx = self.target_idx + 4

        mask[mask == target_idx] = 0
        mask[mask == -1] = 50
        mask[mask != 0] = 1

        obs = np.concatenate([rgba[..., :3], depth[...,None], mask[...,None]], axis=-1)
        obs = self.process_image(obs[...,:3], obs[...,[3]], obs[...,[4]], tuple(self._resize_img_size))
        intrinsic_matrix = projection_to_intrinsics(hand_proj_matrix, self._window_width, self._window_height)
        point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, obs[4].T)   #obs[4].T

        point_state = self.cam_offset[:3,:3].dot(point_state) + self.cam_offset[:3, [3]]
        point_state[1] *= -1
        point_state = self.process_pointcloud(point_state, vis, acc)
        obs = (point_state, obs)
        pose_info = (object_pose, ef_pose)
        return [obs, joint_pos, camera_info, pose_info]

    def retract(self, record=False):
        """
        Move the arm to lift the object.
        """

        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-2:] = 0 # close finger
        observations = [self.step(cur_joint, repeat=300, config=True, vis=False)[0]]
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]

        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                               self._panda.pandaEndEffectorIndex, pos))
            jointPoses[-2:] = 0.0
            obs = self.step(jointPoses, config=True)[0]
            if record:
                observations.append(obs)

        self.retracted = True
        rew = self._reward()
        if record:
            return (rew, observations)
        return rew

    def _reward(self):
        """
        Calculates the reward for the episode.
        """
        reward = 0

        if self.retracted and self.target_lifted():
            print('target {} lifted !'.format(self.target_name))
            reward = 1 #
        return reward


    def _termination(self, depth_img, mask_img, use_depth_heuristics=False):
        """
        Target depth heuristics for determining if grasp can be executed.
        The threshold is based on depth in the middle of the camera and the finger is near the bottom two sides
        """

        depth_heuristics = False
        nontarget_mask = mask_img[...,0] != 0

        if use_depth_heuristics:
            depth_img = depth_img[...,0]
            depth_img[nontarget_mask] = 10
            # hard coded region
            depth_img_roi = depth_img[int(38. * self._window_height / 64):,
            int(24. * self._window_width / 64):int(48 * self._window_width / 64)]
            depth_img_roi_ = depth_img_roi[depth_img_roi < 0.1]
            if depth_img_roi_.shape[0] > 1:
                depth_heuristics = (depth_img_roi_ < 0.045).sum() > 10

        return self._env_step >= self._maxSteps or depth_heuristics or self.target_fall_down()

    def cache_objects(self):
        """
        Load all YCB objects and set up
        """

        obj_path = os.path.join(self.root_dir, 'data/objects/')
        objects = self.obj_indexes
        obj_path = [obj_path + objects[i] for i in self._all_obj]

        self.target_obj_indexes = [self._all_obj.index(idx) for idx in self._target_objs]
        pose = np.zeros([len(obj_path), 3])
        pose[:, 0] = -0.5 - np.linspace(0, 4, len(obj_path))
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() + '/' for p_ in obj_path]
        objectUids = []
        self.object_heights = []
        self.obj_path = objects_paths + self.obj_path
        self.placed_object_poses = []

        for i, name in enumerate(objects_paths):
            trans = pose[i] + np.array(pos) # fixed position
            self.placed_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(os.path.join(self.root_dir, name, 'model_normalized.urdf'), trans, orn)  # xyzw

            if self._change_dynamics:
              p.changeDynamics(uid, -1, lateralFriction=0.15, spinningFriction=0.1, rollingFriction=0.1)

            point_z = np.loadtxt(os.path.join(self.root_dir, name, 'model_normalized.extent.txt'))
            half_height = float(point_z.max()) / 2 if len(point_z) > 0 else 0.01
            self.object_heights.append(half_height)
            objectUids.append(uid)
            p.setCollisionFilterPair(uid, self.plane_id, -1, -1, 0)

            if self._disable_unnece_collision:
              for other_uid in objectUids:
                  p.setCollisionFilterPair(uid, other_uid, -1, -1, 0)
        self.objects_loaded = True
        self.placed_objects = [False] * len(self.obj_path)
        return objectUids


    def cache_reset(self, scene_file, init_joints, enforce_face_target):
        """
        Hack to move the loaded objects around to avoid loading multiple times
        """

        self._panda.reset(init_joints)
        self.place_back_objects()
        if scene_file is None or not os.path.exists(os.path.join(self.data_root_dir, scene_file + '.mat')):
            self._randomly_place_objects(self._get_random_object(self._numObjects), scale=1)
        else:
            self.place_objects_from_scene(scene_file, self._objectUids)

        self._env_step = 0
        self.retracted = False
        self.collided = False
        self.collided_before = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        self.curr_acc_points = np.zeros([3, 0])

        if self._domain_randomization:
            self.load_textures()
            rand_tex_id = np.random.choice(len(self.table_textures))
            p.changeVisualShape(self._objectUids[self.target_idx], -1,
                            textureUniqueId=self.table_textures[rand_tex_id] )
            rand_tex_id = np.random.choice(len(self.table_textures))
            p.changeVisualShape(self._objectUids[-2], -1,
                              textureUniqueId=self.table_textures[rand_tex_id] )
            rand_tex_id = np.random.choice(len(self.table_textures))
            p.changeVisualShape(self._objectUids[-1], -1,
                              textureUniqueId=self.table_textures[rand_tex_id] )

        observation = self.enforce_face_target() if enforce_face_target else self._get_observation()
        observation = self.input_selection(observation)
        return observation

    def place_objects_from_scene(self, scene_file, objectUids=None):
        """
        Place objects with poses based on the scene file
        """

        if self.objects_loaded:
            objectUids = self._objectUids

        scene = sio.loadmat(os.path.join(self.data_root_dir, scene_file + '.mat'))
        poses = scene['pose']
        path = scene['path']

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        new_objs = objectUids is None
        objects_paths = [p_.strip() + '/' for p_ in path]

        for i, name in enumerate(objects_paths[:-2]):
            pose = poses[i]
            trans = pose[:3, 3] + np.array(pos) # fixed position
            orn = ros_quat(mat2quat(pose[:3, :3]))

            full_name = os.path.join(self.root_dir, name)
            if full_name not in self.obj_path:
                continue
            k = self.obj_path.index(full_name) if self.objects_loaded else i
            self.placed_objects[k] = True
            p.resetBasePositionAndOrientation(objectUids[k], trans, orn)
            p.resetBaseVelocity(
            objectUids[k], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )

        rand_name = objects_paths[0]
        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, rand_name))
        self.target_name = rand_name.split('/')[-2]
        print('==== loaded scene: {} target: {} idx: {} init joint'.format(scene_file.split('/')[-1],
                         self.target_name, self.target_idx))

        if 'init_joints' in scene:
            self.reset_joint(scene['init_joints'])
        return objectUids


    def place_back_objects(self):
        for idx, obj in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                p.resetBasePositionAndOrientation(obj, self.placed_object_poses[idx][0], self.placed_object_poses[idx][1])
            self.placed_objects[idx] = False

    def load_textures(self):
        if hasattr(self, 'table_textures'): return
        texture_dir = os.path.join(self.root_dir, 'data/random_textures/textures')
        files = os.listdir(texture_dir)
        random_files = random.sample(files, 200)
        table_textures = [p.loadTexture(os.path.join(texture_dir, f))  for f in random_files ]
        print('number of textures:', len(table_textures))
        self.table_textures = table_textures

    def input_selection(self, observation):
        """
        Select input channels based on data type
        """
        return observation

    def update_curr_acc_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        pos, rot = self._get_ef_pose()
        ef_pose = unpack_pose(np.hstack((pos, tf_quat(rot))))
        new_points = ef_pose[:3,:3].dot(new_points) + ef_pose[:3,[3]]

        # accumulate points
        index = np.random.choice(range(new_points.shape[1]),
                                 size=int(self._pt_accumulate_ratio**self._env_step * new_points.shape[1]), replace=False).astype(np.int)
        self.curr_acc_points = np.concatenate((new_points[:,index], self.curr_acc_points), axis=1) #

    def _get_init_info(self):
        return [self.obj_names, self.obj_poses, np.array(self._panda.getJointStates()[0])]

    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid

    def reset_joint(self, init_joints):
        if init_joints is not None:
          self._panda.reset(np.array(init_joints).flatten())

    def process_action(self, action, delta=False, config=False):
        """
        Process different action types
        """
        if  config:
            if delta:
                cur_joint = np.array(self._panda.getJointStates()[0])
                action = cur_joint + action

        elif self._action_space == 'task6d':
            # transform to local coordinate
            cur_ef = np.array(self._panda.getJointStates()[0])[-3]
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]

            pose = np.eye(4)
            pose[:3, :3] = quat2mat(tf_quat(orn))
            pose[:3, 3] = pos

            pose_delta = np.eye(4)
            pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
            pose_delta[:3, 3] = action[:3]

            new_pose = pose.dot(pose_delta)
            orn = ros_quat(mat2quat(new_pose[:3, :3]))
            pos = new_pose[:3, 3]

            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                  self._panda.pandaEndEffectorIndex, pos, orn))
            jointPoses[-2:] = 0.04
            action = jointPoses
        return action


    def _sample_ef(self, target, near=0.35, far=0.50):
        # sample a camera extrinsics

        count = 0
        ik = None
        outer_loop_num = 20
        inner_loop_num = 5
        if not self._planner_setup :
            try:
                self.setup_expert_scene()
            except:
                pass
        for _ in range(outer_loop_num):
            theta = np.random.uniform(low=0, high=2*np.pi/3)
            phi = np.random.uniform(low=np.pi/2, high=3*np.pi/2) # top sphere
            r = np.random.uniform(low=self._initial_near, high=self._initial_far) # sphere radius
            pos = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

            trans = pos + target + np.random.uniform(-0.03, 0.03, 3)
            trans[2] = np.clip(trans[2], 0.2, 0.6)
            trans[1] = np.clip(trans[1], -0.3, 0.3)
            trans[0] = np.clip(trans[0], 0.0, 0.5)
            pos = trans - target

            for i in range(inner_loop_num):
                rand_up = np.array([0, 0, -1])
                rand_up = rand_up / np.linalg.norm(rand_up)
                R = inv_lookat(pos, 2 * pos, rand_up).dot(rotZ(-np.pi/2)[:3, :3])
                quat = ros_quat(mat2quat(R))
                ik = self.robot.inverse_kinematics(trans, quat, seed=anchor_seeds[np.random.randint(len(anchor_seeds))]) # , quat
                if  ik is not None:
                    break
        return ik

    def randomize_arm_init(self, near=0.35, far=0.50):
        target_forward = self._get_target_relative_pose('base')[:3, 3]
        init_joints = self._sample_ef(target_forward, near=near, far=far)

        if init_joints is not None:
            return list(init_joints) + [0, 0.04, 0.04]
        return None

    def _get_hand_camera_view(self, cam_pose=None):
        """
        Get hand camera view
        """
        if cam_pose is None:
            pos, orn = p.getLinkState(self._panda.pandaUid, 10)[:2]
            cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        fov = 90
        aspect = float(self._window_width) / (self._window_height)
        hand_near = 0.035
        hand_far =  2
        hand_proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, hand_near, hand_far)
        hand_cam_view_matrix = se3_inverse(cam_pose_mat.dot(rotX(-np.pi/2).dot(rotZ(-np.pi)))).T # z backward

        lightDistance = 2.0
        lightDirection = self.table_pos - self._light_position
        lightColor = np.array([1., 1., 1.])
        light_center = np.array([-1.0, 0, 2.5])
        return hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, hand_near, hand_far

    def target_fall_down(self):
        """
        Check if target has fallen down
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height < -0.03:
            return True
        return False

    def target_lifted(self):
        """
        Check if target has been lifted
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height > 0.08:
            return True
        return False

    def setup_expert_scene(self):
        """
        Load all meshes once and then update pose
        """
        # parameters

        self.robot = robot_pykdl.robot_kinematics(None, data_path=self.root_dir + "/")

        print('set up expert scene ...')
        for key, val in self._omg_config.items():
            setattr(planner_cfg, key, val)

        planner_cfg.get_global_param(planner_cfg.timesteps)
        planner_cfg.get_global_path()

        # load obstacles
        self.planner_scene = PlanningScene(planner_cfg)
        self.planner_scene.traj.start = np.array(self._panda.getJointStates()[0])
        self.planner_scene.env.clear()
        obj_names, obj_poses = self.get_env_info(self._cur_scene_file)
        object_lists = [name.split('/')[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        for i, name in enumerate(self.obj_path[:-2]):
            name = name.split('/')[-2]
            trans, orn = self.placed_object_poses[i]
            self.planner_scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

        self.planner_scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1,0,0,0])) # never moved
        self.planner_scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0., 0]))
        self.planner_scene.env.combine_sdfs()
        self._planner_setup = True

    def expert_plan(self, step=-1, return_success=False):
        """
        Run OMG planner for the current scene
        """
        if not self._planner_setup :
            self.setup_expert_scene()
        obj_names, obj_poses = self.get_env_info(self._cur_scene_file)
        object_lists = [name.split('/')[-1].strip() for name in obj_names]

        object_poses = [pack_pose(pose) for pose in obj_poses]
        exists_ids = []
        placed_poses = []
        if self.target_idx == -1 or self.target_name == 'noexists':
            if not return_success:
                return [], np.zeros(0)
            return [], np.zeros(0), False

        for i, name in enumerate(object_lists[:-2]):  # for this scene
            self.planner_scene.env.update_pose(name, object_poses[i])
            idx = self.obj_path[:-2].index(os.path.join(self.root_dir, 'data/objects/' + name + '/'))
            exists_ids.append(idx)
            trans, orn = self.placed_object_poses[idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))

        planner_cfg.disable_collision_set = [name.split('/')[-2] for idx, name in enumerate(self.obj_path[:-2])
                                             if idx not in exists_ids]

        joint_pos = self._panda.getJointStates()[0]
        self.planner_scene.traj.start = np.array(joint_pos)
        self.planner_scene.env.set_target(self.obj_path[self.target_idx].split('/')[-2]) #scene.env.names[0])

        if step > 0: # plan length
            self.planner_scene.env.objects[self.planner_scene.env.target_idx].compute_grasp = False
            planner_cfg.timesteps = step # 20
            planner_cfg.get_global_param(planner_cfg.timesteps)
            self.planner_scene.reset(lazy=True)
            info = self.planner_scene.step()
            planner_cfg.timesteps = self._expert_step # 20
            planner_cfg.get_global_param(planner_cfg.timesteps)
        else:
            self.planner_scene.reset(lazy=True)
            info = self.planner_scene.step()

        plan = self.planner_scene.planner.history_trajectories[-1]
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        ef_pose = unpack_pose(base_pose).dot(self.robot.forward_kinematics_parallel(
                            wrap_value(plan[-1])[None], offset=False)[0][-3]) # world coordinate

        pos, orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx]) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        self.cur_goal = se3_inverse(unpack_pose(obj_pose)).dot(ef_pose)

        for i, name in enumerate(object_lists[:-2]): # reset
            self.planner_scene.env.update_pose(name, placed_poses[i])

        success = info[-1]['terminate'] if len(info) > 1 else False
        if not return_success:
            return plan, np.zeros(len(plan))
        return plan, np.zeros(len(plan)), success


    def _randomly_place_objects(self, urdfList, scale, poses=None):
        """
        Randomize positions of each object urdf.
        """

        xpos = 0.5 + 0.2 * (self._blockRandom * random.random() - 0.5)  - self._shift[0]
        ypos = 0.5 * self._blockRandom * (random.random() - 0.5)  - self._shift[0]
        obj_path = '/'.join(urdfList[0].split('/')[:-1]) + '/'

        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, obj_path))
        self.placed_objects[self.target_idx] = True
        self.target_name = urdfList[0].split('/')[-2]
        x_rot =  0
        z_init = -.65 + 2 * self.object_heights[self.target_idx]
        orn = p.getQuaternionFromEuler([x_rot, 0, np.random.uniform(-np.pi, np.pi)])
        p.resetBasePositionAndOrientation(self._objectUids[self.target_idx], \
                                          [xpos, ypos,  z_init - self._shift[2]], [orn[0], orn[1], orn[2], orn[3]])
        p.resetBaseVelocity(
            self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        for _ in range(2000):
            p.stepSimulation()

        pos, new_orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx]) # to target
        ang = np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1) * 180.0 / np.pi
        print('>>>> target name: {}'.format(self.target_name ))

        if self.target_name in self._filter_objects or ang > 50: # self.target_name.startswith('0') and
            self.target_name = 'noexists'
        return []

    def _load_index_objs(self, file_dir):

        self._target_objs = range(len(file_dir))
        self._all_obj = range(len(file_dir))
        self.obj_indexes = file_dir

    def _get_random_object(self, num_objects):
        """
        Randomly choose an object urdf from the selected objects
        """
        obstacles = self._all_obj
        target_obj = [np.random.randint(0, len(self.obj_indexes))] #
        selected_objects = target_obj
        selected_objects_filenames = [os.path.join('data/objects/', self.obj_indexes[int(selected_objects[0])],
                                                   'model_normalized.urdf')]
        return selected_objects_filenames

    def enforce_face_target(self):
        """
        Move the gripper to face the target
        """
        target_forward = self._get_target_relative_pose('ef')[:3, 3]
        target_forward = target_forward / np.linalg.norm(target_forward)
        r = a2e(target_forward)
        action = np.hstack([np.zeros(3), r])
        return self.step(action, repeat=200, vis=False)[0]

    def random_perturb(self):
        """
        Random perturb
        """
        t = np.random.uniform(-0.04, 0.04, size=(3,))
        r = np.random.uniform(-0.2, 0.2, size=(3,))
        action = np.hstack([t, r])
        return self.step(action, repeat=150, vis=False)[0]

    def collision_check(self):
        """
        Check collision against all links
        """
        if any_link_pair_collision(self._objectUids[self.target_idx], self._panda.pandaUid, CLIENT=self.cid):
            if self._accumulate_points and self.curr_acc_points.shape[1] > self._uniform_num_pts: # touch the target object
                self.curr_acc_points = regularize_pc_point_count(self.curr_acc_points.T, self._uniform_num_pts).T
            self.collided = True
            self.collided_before = True
        else:
            self.collided = False

    def get_env_info(self, scene_file=None):
        """
        Return object names and poses of the current scene
        """

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx] or idx >= len(self._objectUids) - 2:
                pos, orn = p.getBasePositionAndOrientation(uid) # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append('/'.join(self.obj_path[idx].split('/')[:-1]).strip()) # .encode("utf-8")

        return obj_dir, poses

    def convert_action_from_joint_to_cartesian(self, joints, joint_old=None, delta=False):
        """
        Convert joint space action to task space action by fk
        """
        if joint_old is None:
            joint_old = np.array(self._panda.getJointStates()[0])
        if delta:
            joints = joints + joint_old

        ef_pose = self.robot.forward_kinematics_parallel(wrap_value(joint_old)[None], offset=False)[0][-3]
        pos, rot = ef_pose[:3 ,3], ef_pose[:3 ,:3]
        ef_pose_ = self.robot.forward_kinematics_parallel(wrap_value(joints)[None], offset=False)[0][-3]
        rel_pose = se3_inverse(ef_pose).dot(ef_pose_)
        action = np.hstack([rel_pose[:3,3], mat2euler(rel_pose[:3,:3])])

        return action

    def convert_action_from_cartesian_to_joint(self, action):
        """
        Convert task space action to joint space action by ik approximately
        """

        curr_joint = np.array(self._panda.getJointStates()[0])
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        rot = quat2mat(tf_quat(orn)).dot(euler2mat(action[3], action[4], action[5]))
        orn = ros_quat(mat2quat(rot))
        position = rot.dot(action[:3]) # local coordinate
        pos = (pos[0]+position[0], pos[1]+position[1], pos[2]+position[2])

        jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                self._panda.pandaEndEffectorIndex, pos, orn))
        jointPoses[-2:] = 0.04
        return np.subtract(jointPoses, curr_joint)

    def process_image(self, color, depth, mask, size=None):
        """
        Normalize RGBDM
        """
        color = color.astype(np.float32) / 255.0
        mask  = mask.astype(np.float32)
        depth = depth.astype(np.float32) / 5000
        if size is not None:
            color = cv2.resize(color, size)
            mask  = cv2.resize(mask, size)
            depth = cv2.resize(depth, size)
        obs = np.concatenate([color, depth[...,None], mask[...,None]], axis=-1)
        obs = obs.transpose([2, 1, 0])
        return obs

    def process_pointcloud(self, point_state, vis, acc_pt=True, use_farthest_point=False):
        """
        Process point cloud input
        """
        if self._accumulate_points and acc_pt:
            self.update_curr_acc_points(point_state)
            pos, rot = self._get_ef_pose()
            ef_pose = se3_inverse(unpack_pose(np.hstack((pos, tf_quat(rot)))))
            point_state = ef_pose[:3,:3].dot(self.curr_acc_points) + ef_pose[:3,[3]]

        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            point_state = regularize_pc_point_count(point_state.T, self._uniform_num_pts, use_farthest_point).T

        if self._use_hand_finger_point:
            point_state = np.concatenate([self.hand_finger_points, point_state], axis=1)
            point_state_ = np.zeros((4, point_state.shape[1]))
            point_state_[:3] = point_state
            point_state_[3, :self.hand_finger_points.shape[1]] = 1
            point_state = point_state_
        if vis:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(point_state.T[:, :3])
            o3d.visualization.draw_geometries([pred_pcd])

        return point_state

    def _get_relative_ef_pose(self):
        """
        Get all obejct poses with respect to the end effector
        """
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid) # to target
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, ef_pose))
        return poses

    def _get_goal_dist(self):
        """
        point distance to goal
        """
        if hasattr(self, 'cur_goal'):
            goal_pose = unpack_pose_rot_first(self._get_relative_goal_pose())
            goal_control_point = goal_pose[:3, :3].dot(self.hand_finger_points) + goal_pose[:3, [3]]
            dist = np.abs(goal_control_point - self.hand_finger_points).sum(-1).mean()
            return  dist
        return 0

    def _get_nearest_goal_pose(self, rotz=False, mat=False):
        """
        Nearest goal query
        """
        curr_joint = np.array(self._panda.getJointStates()[0])
        goal_set  = self.planner_scene.traj.goal_set
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]

        pos, orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx]) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        ws_goal_set = self.planner_scene.env.objects[self.planner_scene.env.target_idx].grasps_poses
        grasp_set_pose = np.matmul(unpack_pose(obj_pose)[None], ws_goal_set)
        rel_pose = np.matmul(se3_inverse(unpack_pose(ef_pose))[None], grasp_set_pose)

        point_1 = self.hand_finger_points
        point_2 = np.matmul(rel_pose[:, :3, :3], self.hand_finger_points[None]) + rel_pose[:, :3, [3]]
        pt_argmin = np.sum(np.abs(point_1[None] - point_2), axis=1).mean(-1).argmin()
        goal_pose = grasp_set_pose[pt_argmin]
        cur_goal = pack_pose(goal_pose)
        self.cur_goal = se3_inverse(unpack_pose(obj_pose)).dot(goal_pose)

        if mat:
          return inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)) if rotz else inv_relative_pose(cur_goal, ef_pose)
        if rotz:
          return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)))
        return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose))

    def _get_relative_goal_pose(self, rotz=False, mat=False, nearest=False):
        """
        Get the relative pose from current to the goal
        """

        if nearest and not self.collided_before: return self._get_nearest_goal_pose(rotz, mat)
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        pos, orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx]) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cur_goal_mat = unpack_pose(obj_pose).dot(self.cur_goal)
        cur_goal = pack_pose(cur_goal_mat)
        if mat:
            return inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)) if rotz else inv_relative_pose(cur_goal, ef_pose)
        if rotz:
            return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)))
        return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose))   # to be compatible with graspnet

    def _get_ef_pose(self, mat=False):
        """
        end effector pose in world frame
        """
        if not mat:
            return p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]])

    def _get_target_relative_pose(self, option='base'):
        """
        Get target obejct poses with respect to the different frame.
        """
        if option == 'base':
            pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        elif option == 'ef':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        elif option == 'tcp':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
            rot = quat2mat(tf_quat(orn))
            tcp_offset = rot.dot(np.array([0,0,0.13]))
            pos = np.array(pos) + tcp_offset

        pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        uid = self._objectUids[self.target_idx]
        pos, orn = p.getBasePositionAndOrientation(uid) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        return inv_relative_pose(obj_pose, pose)



if __name__ == '__main__':
    pass
