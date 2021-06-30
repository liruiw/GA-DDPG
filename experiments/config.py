# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Tuning online training scale based on local GPU and memory limits. The code is test on 4
V100 GPU, and 100 GB CPU memory. 2 GPUs are used for actor rollout and the other two for training.

The configs that can be adjusted:
num_remotes, batch_size, RL_MEMORY_SIZE, @ray.remote(num_cpus=*, num_gpus=*)
"""

import os
import os.path as osp
import numpy as np
import math
import tabulate
from easydict import EasyDict as edict
import IPython
import yaml

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# create output folders
if not os.path.exists(os.path.join(root_dir, 'output_misc')):
    os.makedirs(os.path.join(root_dir, 'output_misc/rl_output_stat'))
if not os.path.exists(os.path.join(root_dir, 'output')):
    os.makedirs(os.path.join(root_dir, 'output'))
if not os.path.exists(os.path.join(root_dir, 'experiments/runs')):
    os.makedirs(os.path.join(root_dir,  'experiments/runs'))

__C = edict()
cfg = __C

# Global options
#
__C.script_name = ''
__C.RNG_SEED = 3
__C.EPOCHS = 200
__C.ROOT_DIR = root_dir + '/'
__C.DATA_ROOT_DIR = 'data/scenes'
__C.ROBOT_DATA_DIR = 'data/robots'
__C.OBJECT_DATA_DIR = 'data/objects'
__C.OUTPUT_DIR = 'output'
__C.OUTPUT_MISC_DIR = 'output_misc'
__C.MODEL_SPEC_DIR = "experiments/model_spec"
__C.EXPERIMENT_OBJ_INDEX_DIR = "experiments/object_index"
__C.LOG = True
__C.IMG_SIZE = (112, 112)

__C.RL_IMG_SIZE = (112, 112)
__C.RL_MAX_STEP = 20
__C.RL_DATA_ROOT_DIR = __C.DATA_ROOT_DIR
__C.RL_SAVE_DATA_ROOT_DIR = 'data/offline_data'
__C.SCRIPT_FOLDER = 'experiments/cfgs'
__C.RL_SAVE_DATA_NAME = 'data_50k.npz'
__C.ONPOLICY_MEMORY_SIZE = -1
__C.RL_MEMORY_SIZE = 100000
__C.OFFLINE_RL_MEMORY_SIZE = 100000
__C.OFFLINE_BATCH_SIZE = 100
__C.pretrained_time = ''
__C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR, 'rl_pointnet_model_spec.yaml')
__C.RL_TEST_SCENE = 'data/gaddpg_scenes'


# RL options
#
__C.RL_TRAIN = edict()

# architecture and network hyperparameter
__C.RL_TRAIN.clip_grad = 0.5
__C.RL_TRAIN.gamma = 0.95
__C.RL_TRAIN.batch_size = 256
__C.RL_TRAIN.updates_per_step = 4
__C.RL_TRAIN.hidden_size = 256

__C.RL_TRAIN.tau = 0.0001
__C.RL_TRAIN.lr = 3e-4
__C.RL_TRAIN.reinit_lr = 1e-4
__C.RL_TRAIN.value_lr = 3e-4
__C.RL_TRAIN.lr_gamma = 0.5
__C.RL_TRAIN.value_lr_gamma = 0.5
__C.RL_TRAIN.head_lr = 3e-4
__C.RL_TRAIN.feature_input_dim = 512
__C.RL_TRAIN.ddpg_coefficients = [0., 0., 1.0, 1., 0.2]
__C.RL_TRAIN.value_milestones = [20000, 40000, 60000, 80000]
__C.RL_TRAIN.policy_milestones = [20000, 40000, 60000, 80000]
__C.RL_TRAIN.mix_milestones = [4000, 8000, 20000, 40000, 60000, 80000, 100000, 140000, 180000]
__C.RL_TRAIN.mix_policy_ratio_list = [0.1, 0.2]
__C.RL_TRAIN.mix_value_ratio_list = [1.]
__C.RL_TRAIN.policy_extra_latent = -1
__C.RL_TRAIN.critic_extra_latent = -1
__C.RL_TRAIN.save_epoch = [5000, 20000, 40000, 80000, 140000, 180000, 200000]
__C.RL_TRAIN.overwrite_feat_milestone = []
__C.RL_TRAIN.fix_timestep_test = True
__C.RL_TRAIN.load_buffer = False
__C.RL_TRAIN.new_scene = True

# algorithm hyperparameter
__C.RL_TRAIN.train_value_feature = True
__C.RL_TRAIN.train_feature = True
__C.RL_TRAIN.reinit_optim = False
__C.RL_TRAIN.off_policy = True
__C.RL_TRAIN.use_action_limit = True
__C.RL_TRAIN.sa_channel_concat = True
__C.RL_TRAIN.use_image = False
__C.RL_TRAIN.dagger = False
__C.RL_TRAIN.concat_option = 'point_wise'
__C.RL_TRAIN.use_time = True
__C.RL_TRAIN.RL = True
__C.RL_TRAIN.value_model = False
__C.RL_TRAIN.shared_feature = False
__C.RL_TRAIN.policy_update_gap = 2
__C.RL_TRAIN.self_supervision = False
__C.RL_TRAIN.critic_goal = False
__C.RL_TRAIN.policy_aux = True
__C.RL_TRAIN.train_goal_feature = False
__C.RL_TRAIN.critic_aux = True
__C.RL_TRAIN.policy_goal = False
__C.RL_TRAIN.goal_reward_flag = False
__C.RL_TRAIN.bc_reward_flag = False
__C.RL_TRAIN.online_buffer_ratio = 0.
__C.RL_TRAIN.onpolicy = False
__C.RL_TRAIN.use_point_state = True
__C.RL_TRAIN.channel_num = 5
__C.RL_TRAIN.refill_buffer = True
__C.RL_TRAIN.change_dynamics = False
__C.RL_TRAIN.pt_accumulate_ratio = 0.95
__C.RL_TRAIN.dart = True
__C.RL_TRAIN.accumulate_points = True
__C.RL_TRAIN.max_epoch = 150000
__C.RL_TRAIN.action_noise = 0.01

# environment hyperparameter
__C.RL_TRAIN.load_obj_num = 40
__C.RL_TRAIN.reinit_factor = 3
__C.RL_TRAIN.shared_objects_across_worker = False
__C.RL_TRAIN.target_update_interval = 3000
__C.RL_TRAIN.env_num_objs = 1
__C.RL_TRAIN.index_split = 'train'
__C.RL_TRAIN.env_name = 'PandaYCBEnv'
__C.RL_TRAIN.index_file = os.path.join(__C.EXPERIMENT_OBJ_INDEX_DIR, 'extra_shape.json')
__C.RL_TRAIN.max_num_pts = 20000
__C.RL_TRAIN.uniform_num_pts = 1024
__C.RL_TRAIN.use_expert_plan = False

# exploration worker hyperparameter
__C.RL_TRAIN.num_remotes = 8
__C.RL_TRAIN.init_distance_low = 0.15
__C.RL_TRAIN.init_distance_high = 0.45
__C.RL_TRAIN.explore_ratio = 0.1
__C.RL_TRAIN.explore_cap = 0.5
__C.RL_TRAIN.explore_ratio_list = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
__C.RL_TRAIN.noise_ratio_list = [3., 2.5, 2., 1.5, 1, 0.5]
__C.RL_TRAIN.noise_type = 'uniform'
__C.RL_TRAIN.expert_initial_state = True
__C.RL_TRAIN.DAGGER_MIN_STEP = 5
__C.RL_TRAIN.DAGGER_MAX_STEP = 18
__C.RL_TRAIN.DAGGER_RATIO = 0.5
__C.RL_TRAIN.DART_MIN_STEP = 5
__C.RL_TRAIN.DART_MAX_STEP = 13
__C.RL_TRAIN.DART_RATIO = 0.5
__C.RL_TRAIN.ENV_RESET_TRIALS = 7
__C.RL_TRAIN.SAVE_EPISODE_INTERVAL = 50
__C.RL_TRAIN.EXPERT_INIT_MIN_STEP = 0
__C.RL_TRAIN.EXPERT_INIT_MAX_STEP = 15
__C.RL_TRAIN.ENV_NEAR = 0.2
__C.RL_TRAIN.ENV_FAR  = 0.5

# misc hyperparameter
__C.RL_TRAIN.load_test_scene_new = False
__C.RL_TRAIN.load_scene_joint = False
__C.RL_TRAIN.log = True
__C.RL_TRAIN.visdom = True
__C.RL_TRAIN.domain_randomization = False
__C.RL_TRAIN.buffer_full_size = -1
__C.RL_TRAIN.buffer_start_idx = 0
__C.RL_TRAIN.fill_data_step   = 10


def process_cfg(reset_model_spec=True):
    """
    hacks to change configs
    """
    if __C.RL_TRAIN.onpolicy and __C.RL_TRAIN.RL:
        __C.RL_TRAIN.explore_cap = 1.0

    if __C.RL_TRAIN.self_supervision and __C.RL_TRAIN.RL:
        __C.RL_TRAIN.expert_initial_state = False
        __C.RL_TRAIN.explore_ratio = 1.0
        __C.RL_TRAIN.action_noise = 0.0

    if __C.RL_TRAIN.use_image:  # image based
        __C.RL_TRAIN.domain_randomization = True

    if reset_model_spec:
        if not __C.RL_TRAIN.use_image:
            __C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR , "rl_pointnet_model_spec.yaml"
            )
        else:
            __C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR , "rl_resnet_model_spec.yaml"
            )
    if __C.RL_TRAIN.sa_channel_concat:
        __C.RL_TRAIN.value_model = True
    if __C.RL_TRAIN.policy_goal:
        __C.RL_TRAIN.train_goal_feature = True

    __C.omg_config = {
        'traj_init':'grasp',
        'scene_file': '' ,
        'vis': False,
        'increment_iks': True ,
        'terminate_smooth_loss': 3 ,
        'ol_alg' :'Proj', #
        'pre_terminate': True ,
        'extra_smooth_steps': 5,
        'traj_interpolate' :"linear",
        'goal_idx' :-1,
        'traj_delta': 0.05,
        'standoff_dist': 0.08,
        'allow_collision_point': 0,
        'clearance' :0.03,
        'ik_clearance': 0.07,
        'smoothness_base_weight' :3,
        'base_obstacle_weight': 1.,
        'target_hand_filter_angle': 90,
        'target_obj_collision': 1,
        'target_epsilon': 0.06,
        'optim_steps' :1,
        'ik_parallel' :False,
        'ik_seed_num' :13,
        'traj_max_step': int(__C.RL_MAX_STEP) + 6,
        'root_dir':  root_dir + "/",
        'traj_min_step' :int(__C.RL_MAX_STEP) - 5,
        'timesteps': int(__C.RL_MAX_STEP),
        'dynamic_timestep': False,
        'use_expert_plan': __C.RL_TRAIN.use_expert_plan,
        'silent': True,
    }

    __C.env_config = {
        "action_space": 'task6d',
        "data_type": 'RGBDM',
        "expert_step": int(__C.RL_MAX_STEP),  # 1.5 *
        "numObjects": 1,
        "width": __C.RL_IMG_SIZE[0],
        "height": __C.RL_IMG_SIZE[1],
        "img_resize": __C.RL_IMG_SIZE,
        "random_target": True,
        "use_hand_finger_point": True,
        "accumulate_points": __C.RL_TRAIN.accumulate_points,
        "uniform_num_pts": __C.RL_TRAIN.uniform_num_pts,
        "regularize_pc_point_count": True,
        "domain_randomization": __C.RL_TRAIN.domain_randomization,
        "change_dynamics": __C.RL_TRAIN.change_dynamics,
        "pt_accumulate_ratio": __C.RL_TRAIN.pt_accumulate_ratio,
        "omg_config": __C.omg_config,
        'initial_near': __C.RL_TRAIN.ENV_NEAR,
        'initial_far':  __C.RL_TRAIN.ENV_FAR,
    }


def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, "output", __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        if not k in b.keys():
            continue

        # the types must match, too
        if type(b[k]) is not type(v):
            continue

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename=None, dict=None, reset_model_spec=True):
    """Load a config file and merge it into the default options."""

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f))
    if not reset_model_spec:
        output_dir = "/".join(filename.split("/")[:-1])
        __C.RL_MODEL_SPEC = os.path.join(
            output_dir, yaml_cfg["RL_MODEL_SPEC"].split("/")[-1]
        )
    if dict is None:
        _merge_a_into_b(yaml_cfg, __C)
    else:
        _merge_a_into_b(yaml_cfg, dict)
    process_cfg(reset_model_spec=reset_model_spec)

def save_cfg_to_file(filename, cfg):
    """Load a config file and merge it into the default options."""
    with open(filename, 'w+') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def cfg_repr(cfg, fmt='plain'):
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items(), tablefmt=fmt)
    return helper(cfg)
