# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime
import numpy as np
import itertools

import torch
from core.bc import BC
from core.ddpg import DDPG
from core.replay_memory import BaseMemory as ReplayMemory
from core.utils import *
from core.trainer import *

from tensorboardX import SummaryWriter
from env.panda_scene import PandaYCBEnv, PandaTaskSpace6D
from experiments.config import *

import json
import time
from collections import deque
import tabulate
import scipy.io as sio
import IPython
import pprint
import glob
import ray
import yaml
import random
import psutil
import GPUtil

def create_parser():
    parser = argparse.ArgumentParser(description='Train Online Args')
    parser.add_argument('--env-name', default="PandaYCBEnv" )
    parser.add_argument('--policy', default="SAC", )
    parser.add_argument('--seed', type=int, default=233, metavar='N')
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained', type=str, default=None, help='use a pretrained model')

    parser.add_argument('--log', action="store_true", help='log loss')
    parser.add_argument('--model_surfix',  type=str, default='latest')
    parser.add_argument('--save_buffer', action="store_true")
    parser.add_argument('--save_online_buffer', action="store_true")
    parser.add_argument('--finetune', action="store_true" )

    parser.add_argument('--config_file', type=str, default="bc.yaml")
    parser.add_argument('--visdom', action="store_true")
    parser.add_argument('--max_load_scene_num', type=int, default=-1)
    parser.add_argument('--load_buffer', action="store_true")
    parser.add_argument('--load_online_buffer', action="store_true", help='load online buffer')
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--save_scene', action="store_true")
    parser.add_argument('--load_scene', action="store_true")
    parser.add_argument('--pretrained_policy_name', type=str, default='BC')

    return parser




def sample_experiment_objects():
    """
    Sample objects from the json files for replay buffer and environment
    """
    index_file = CONFIG.index_file.split('.json')[0].split('/')[-1]
    index_file = os.path.join(cfg.EXPERIMENT_OBJ_INDEX_DIR, index_file + '.json')

    file_index = json.load(open(index_file))[CONFIG.index_split]
    file_dir = [f[:-5].split('.')[0][:-2] if 'json' in f else f for f in file_index ]
    sample_index = np.random.choice(range(len(file_dir)), min(LOAD_OBJ_NUM, len(file_dir)), replace=False).astype(np.int)
    file_dir = [file_dir[idx] for idx in sample_index]

    file_dir = list(set(file_dir))
    print('training object index: {} obj num: {}'.format(index_file, len(file_dir)))

    return file_dir

def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """
    if args.fix_output_time is None:
        dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    else:
        dt_string = args.fix_output_time

    model_output_dir = os.path.join(cfg.OUTPUT_DIR, dt_string)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    os.makedirs(model_output_dir)
    load_from_pretrain = args.pretrained is not None and os.path.exists(args.pretrained)

    if load_from_pretrain:
        """ load pretrained config and copy model """
        cfg_folder = args.pretrained
        if os.path.exists(os.path.join(cfg_folder, "config.yaml")):
            cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)
        cfg.pretrained_time = args.pretrained.split("/")[-1]
        migrate_model(
            args.pretrained,
            model_output_dir,
            args.model_surfix,
            False,
        )

    if args.config_file is not None:
        """ overwrite and store new config  """
        script_file = os.path.join(cfg.SCRIPT_FOLDER, args.config_file)
        if os.path.exists(script_file):
            cfg_from_file(script_file)
        cfg.script_name = args.config_file
        os.system(
            "cp {} {}".format(
                script_file, os.path.join(model_output_dir, args.config_file)
            )
        )

    os.system(
        "cp {} {}".format(
            cfg.RL_MODEL_SPEC,
            os.path.join(model_output_dir, cfg.RL_MODEL_SPEC.split("/")[-1]),
        )
    )
    save_cfg_to_file(os.path.join(model_output_dir, "config.yaml"), cfg)

    return dt_string


class ActorWrapper(object):
    """
    Wrapper class for actors to do rollouts and save data, to collect data while training
    """
    def __init__(self,  learner_id, buffer_remote_id, online_buffer_remote_id, unique_id):
        from env.panda_scene import PandaYCBEnv
        self.learner_id = learner_id
        self.buffer_id = buffer_remote_id
        self.unique_id = unique_id
        self.online_buffer_id = online_buffer_remote_id

        self.env = eval(CONFIG.env_name)(**cfg.env_config)
        self.target_obj_list = []
        np.random.seed(args.seed + unique_id)
        objects = sample_experiment_objects() if not CONFIG.shared_objects_across_worker else CONFIG.sampled_objs

        self.env._load_index_objs(objects) # CONFIG.sampled_objs
        self.env.reset(save=False, data_root_dir=cfg.DATA_ROOT_DIR, cam_random=0,
                                   enforce_face_target=True)

        if VISDOM:
            self.vis = Visdom(port=8097)
            self.win_id = self.vis.image(np.zeros([3, int(cfg.RL_IMG_SIZE[0]), int(cfg.RL_IMG_SIZE[1])]))
        self._TOTAL_CNT, self._TOTAL_REW = 1, 0
        self.offset_pose, self.K = get_camera_constant(cfg.RL_IMG_SIZE[0])

    def reset_env(self):
        """
        reset the environment by loading new objects
        """
        from env.panda_scene import PandaYCBEnv
        self.env = eval(CONFIG.env_name)(**cfg.env_config)
        objects = sample_experiment_objects()  if not CONFIG.shared_objects_across_worker else CONFIG.sampled_objs
        self.env._load_index_objs(objects)
        self.env.reset( save=False, data_root_dir=cfg.DATA_ROOT_DIR,
                        cam_random=0, enforce_face_target=True)

    def init_episode(self):
        """
        Initialize an episode by sampling objects and init states until valid
        """
        check_scene_flag = False
        data_root = cfg.DATA_ROOT_DIR
        scenes = None
        state = self.env.reset( save=False, scene_file=scenes,
                                data_root_dir=data_root, cam_random=0, reset_free=True,
                                enforce_face_target=True)
        if VISDOM and state is not None:  self.vis.image(state[0][1][:3].transpose([0,2,1]), win=self.win_id)
        init_joints = None
        for i in range(ENV_RESET_TRIALS):
            init_joints = rand_sample_joint(self.env, init_joints)
            if init_joints is not None:
                self.env.reset_joint(init_joints)
                start_rotation = self.env._get_ef_pose(mat=True)[:3, :3]
                if check_scene(self.env, state, start_rotation):
                    check_scene_flag = True
                    break
        return state, check_scene_flag

    def get_flags(self, explore, expert_traj_length, step):
        """
        get different booleans for the current step
        """
        expert_flag   = float(not explore)
        perturb_flags = 0
        apply_dagger  = CONFIG.dagger and \
                        (step > DAGGER_MIN_STEP) and \
                        (step < min(DAGGER_MAX_STEP, expert_traj_length-8)) and \
                        (np.random.uniform() < DAGGER_RATIO) and explore
        apply_dart    = CONFIG.dart and \
                        (step > CONFIG.DART_MIN_STEP) and \
                        (step < CONFIG.DART_MAX_STEP) and \
                        (np.random.uniform() < CONFIG.DART_RATIO) and not explore
        return expert_flag, perturb_flags, apply_dagger, apply_dart

    def rollout(self, num_episodes=1, explore=False, dagger=False,
                      test=False,  noise_scale=1.):
        """
        policy rollout and save data
        """
        for _ in range(num_episodes):

            # init scene
            try:
                state, check_scene_flag = self.init_episode( )
            except:
                print('init episode error')
                check_scene_flag = False
            if not check_scene_flag:  return [0]

            step, reward = 0., 0.
            done = False
            cur_episode = []
            expert_plan, omg_cost = self.env.expert_plan()
            expert_traj_length = len(expert_plan)
            if expert_traj_length >= EXTEND_MAX_STEP or expert_traj_length < 5 or state is None:
                return [0]

            init_info = self.env._get_init_info()
            expert_initial_step = np.random.randint(EXPERT_INIT_MIN_STEP, EXPERT_INIT_MAX_STEP)
            expert_initial = CONFIG.expert_initial_state and not test and not BC
            goal_involved = CONFIG.train_goal_feature or CONFIG.policy_aux  or CONFIG.critic_aux
            aux_pred = np.zeros(0)

            # rollout
            while not done:

                # plan
                expert_flag, perturb_flags, apply_dagger, apply_dart = self.get_flags(explore, expert_traj_length, step)
                if apply_dart:
                    perturb_flags = 1.
                    self.env.random_perturb() # inject noise

                if  apply_dagger:
                    expert_flag = 2.

                if apply_dagger or apply_dart:  # replan
                    rest_expert_plan, _ = self.env.expert_plan(step=int(MAX_STEP-step-1))
                    expert_plan = np.concatenate((expert_plan[:int(step)], rest_expert_plan), axis=0)
                    expert_traj_length = len(expert_plan)

                goal_pose = self.env._get_relative_goal_pose(nearest=explore and not apply_dagger)
                if step < len(expert_plan): expert_joint_action = expert_plan[int(step)]
                expert_action = self.env.convert_action_from_joint_to_cartesian(expert_joint_action)

                # expert
                if not explore or (expert_initial and step < expert_initial_step):
                    grasp = step == len(expert_plan) - 1
                    action = expert_action
                    log_probs = np.zeros(6)

                # agent
                else:
                    remain_timestep = max(expert_traj_length-step, 1)
                    action_mean, log_probs, action_sample, aux_pred = ray.get(self.learner_id.select_action.remote(state,
                            goal_state=goal_pose, remain_timestep=remain_timestep,
                            gt_goal_rollout=not CONFIG.self_supervision and not test))
                    noise = get_noise_delta(action_mean, CONFIG.action_noise, CONFIG.noise_type)
                    action_mean = action_mean + noise * noise_scale
                    action = action_mean
                    grasp = 0

                # step
                next_state, reward, done, _ = self.env.step(action, delta=True)
                if VISDOM:
                    img = draw_grasp_img(next_state[0][1][:3].transpose([2,1,0]), unpack_pose_rot_first(goal_pose),
                                         self.K,  self.offset_pose, (0, 1., 0))
                    if goal_involved and len(aux_pred) == 7:
                        img = draw_grasp_img(next_state[0][1][:3].transpose([2,1,0]), unpack_pose_rot_first(aux_pred),
                                         self.K,  self.offset_pose, (0, 1., 0))
                    self.vis.image(img.transpose([2,0,1]), win=self.win_id)

                if (not explore and step == expert_traj_length - 1) or step == EXTEND_MAX_STEP or (done):
                    reward, res_obs = self.env.retract(record=True)
                    if VISDOM:
                        for r in res_obs:  self.vis.image(r[0][1][:3].transpose([0,2,1]), win=self.win_id)  #
                    done = True

                step_dict = {
                                'point_state': state[0][0],
                                'image_state': state[0][1][None],
                                'expert_action': expert_action[None],
                                'reward': reward,
                                'returns': reward,
                                'terminal': done,
                                'timestep': step,
                                'pose': state[2],
                                'target_pose': state[-1][0],
                                'state_pose': state[-1][1],
                                'target_idx': self.env.target_idx,
                                'target_name': self.env.target_name,
                                'collide': self.env.collided,
                                'expert_flags': expert_flag,
                                'perturb_flags': perturb_flags,
                                'grasp': grasp,
                                'goal': goal_pose,
                                'action': action[None]
                             }

                cur_episode.append(step_dict)
                step = step + 1.
                state = next_state

            reward = reward > 0.5
            if ON_POLICY and explore: # separate BC and RL
                self.online_buffer_id.add_episode.remote(cur_episode, explore, test)
            else:
                self.buffer_id.add_episode.remote(cur_episode,  explore, test)

        return [reward]



# adjust num_gpus
@ray.remote(num_cpus=1, num_gpus=0.12)
class ActorWrapper008(ActorWrapper):
    pass




def reinit(reset=False):
    print_and_write(None, '============================ Reinit ==========================')
    time.sleep(4)
    CONFIG.sampled_objs = sample_experiment_objects()
    rollouts = [actor.reset_env.remote() for i, actor in enumerate(actors)]
    res = ray.get(rollouts)
    gpu_usage, memory_usage = get_usage()
    gpu_max = float(gpu_usage) / gpu_limit > 0.98
    memory_max = memory_usage >= MEMORY_THRE
    print('==================== Memory: {} GPU: {} ====================='.format(memory_usage, gpu_usage))

    if  reset:
        os.system('nvidia-smi')
        print_and_write(None, '===================== Ray Reinit =================')
        ray.get(learner_id.save_model.remote())
        time.sleep(10)
        ray.shutdown()
        time.sleep(2)
        return get_ray_objects(reinit=True)

    print_and_write(None, '==============================================================')



def get_ray_objects(reinit=False):
    rollout_agent_wrapper = RolloutAgentWrapperGPU1
    gpu_usage, memory_usage = get_usage()
    print('==================== Reset Memory: {} GPU: {} ====================='.format(memory_usage, gpu_usage))

    ray.init(num_cpus=5 * NUM_REMOTES + 6, object_store_memory=object_store_memory, webui_host="0.0.0.0")
    buffer_id = ReplayMemoryWrapper.remote(int(cfg.RL_MEMORY_SIZE), cfg, 'expert')
    if LOAD_MEMORY:
        ray.get(buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, int(cfg.RL_MEMORY_SIZE)))
    if ON_POLICY:
        buffer_size = cfg.ONPOLICY_MEMORY_SIZE if cfg.ONPOLICY_MEMORY_SIZE > 0 else cfg.RL_MEMORY_SIZE
        online_buffer_id = ReplayMemoryWrapper.remote(int(buffer_size), cfg, 'online')
        if args.load_online_buffer:
            ray.get(online_buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, int(buffer_size) ))
    else:
        online_buffer_id = ReplayMemoryWrapper.remote(100,  cfg, 'online') # dummy

    if reinit:
        learner_id = agent_wrapper.remote(args, cfg, init_pretrained_path,
                                          input_dim, logdir, True, args.model_surfix, model_output_dir)
        rollout_agent_ids = [rollout_agent_wrapper.remote(args, cfg,  init_pretrained_path,
                                          input_dim, None, True, args.model_surfix, model_output_dir) ]
    else:
        learner_id = agent_wrapper.remote(args, cfg, pretrained_path, input_dim, None)
        rollout_agent_ids = [rollout_agent_wrapper.remote(args, cfg, init_pretrained_path,
                                           input_dim, None, True, args.model_surfix, model_output_dir) ]

    trainer = TrainerRemote.remote(args, cfg, learner_id, buffer_id, online_buffer_id, logdir, model_output_dir)
    CONFIG.sampled_objs = sample_experiment_objects()

    actors =  [actor_wrapper.remote(rollout_agent_ids[0], buffer_id, online_buffer_id, actor_idx) for actor_idx in range(NUM_REMOTES)]
    return actors, rollout_agent_ids, learner_id, trainer, buffer_id, online_buffer_id

def get_buffer_log():
    """Get gpu and memory usages as well as current performance """
    reward_info, online_reward_info = np.array(ray.get(buffer_id.reward_info.remote())), np.array(ray.get(online_buffer_id.reward_info.remote()))
    return [(reward_info[i], online_reward_info[i]) for i in range(len(reward_info))]


def log_info():
    actor_name = 'ONLINE' if explore else 'EXPERT'
    rollout_time = time.time() - start_rollout
    gpu_usage, memory_usage = get_usage()
    print_and_write(None, '===== Epoch: {} | Actor: {} | Worker: {} | Explore: {:.4f}  ======'.format(
                                                reinit_count, actor_name, NUM_REMOTES, explore_ratio  ))
    print_and_write(None, ( 'TIME: {:.2f} MEMORY: {:.1f} GPU: {:.0f}  REWARD {:.3f}/{:.3f} ' + \
                            'COLLISION {:.3f}/{:.3f} SUCCESS {:.3f}/{:.3f}\n' + \
                            'DATE: {} BATCH: {}').format(
                            rollout_time, memory_usage, gpu_usage,  buffer_log[1][0],
                            buffer_log[1][1], buffer_log[4][0], buffer_log[4][1],
                            buffer_log[5][0], buffer_log[5][1],
                            datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"), CONFIG.batch_size))
    print_and_write(None, '===========================================================================')
    gpu_max = (float(gpu_usage) / gpu_limit) > 0.98
    memory_max = memory_usage >= MEMORY_THRE
    iter_max = (train_iter + 4) % (reinit_interval) == 0
    return gpu_max, memory_max, iter_max

def choose_setup():
    NUM_REMOTES =  CONFIG.num_remotes
    agent_wrapper = AgentWrapperGPU1
    actor_wrapper = ActorWrapper008
    GPUs = GPUtil.getGPUs()
    max_memory = 25

    if len(GPUs) == 1: # 4 GPU
        NUM_REMOTES //= 2
        agent_wrapper = AgentWrapperGPU05 # 2

    if len(GPUs) == 4 and CLUSTER: # 4 GPU
        CONFIG.batch_size = int(CONFIG.batch_size * 2)
        NUM_REMOTES = int(NUM_REMOTES * 2)
        agent_wrapper = AgentWrapperGPU2 # 2

    print('update batch size: {} worker: {} memory: {}'.format(CONFIG.batch_size, NUM_REMOTES, max_memory))
    return actor_wrapper, agent_wrapper, max_memory, NUM_REMOTES

def start_log():
    logdir = '{}/{}/{}_{}'.format(cfg.OUTPUT_DIR,output_time, CONFIG.env_name, POLICY)
    CONFIG.output_time = output_time
    CONFIG.model_output_dir = model_output_dir
    CONFIG.logdir = logdir
    CONFIG.CLUSTER = CLUSTER
    CONFIG.ON_POLICY = ON_POLICY
    pretrained_path = os.path.join(cfg.OUTPUT_DIR, output_time)
    init_pretrained_path = pretrained_path
    print('output_time: {} logdir: {}'.format(output_time, logdir))
    return pretrained_path, logdir, init_pretrained_path


def get_usage_and_success():
    """Get gpu and memory usages as well as current performance """
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    reward_info, online_reward_info = np.array(ray.get(buffer_id.reward_info.remote())), np.array(ray.get(online_buffer_id.reward_info.remote()))
    total_success, success, onpolicy_success, test_success = reward_info
    total_online_success, online_success, online_onpolicy_success, online_test_success = online_reward_info

    return gpu_usage, memory_usage, (online_success, success), (online_onpolicy_success, onpolicy_success), \
                (total_online_success, total_success), (online_test_success, test_success)


def copy_tensorboard_log():
    """copy tensorboard log """
    if os.path.isdir(logdir):
        os.system('cp -r {} {}'.format(logdir, model_output_dir))


if __name__ == "__main__":
    # config
    parser = create_parser()
    args = parser.parse_args()
    BC  = 'BC' in args.policy
    POLICY = args.policy
    CONFIG = cfg.RL_TRAIN
    CONFIG.RL = False if BC else True
    output_time = setup()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    MAX_STEP = cfg.RL_MAX_STEP
    LOAD_OBJ_NUM = CONFIG.load_obj_num
    DAGGER_RATIO = CONFIG.DAGGER_RATIO
    SAVE_EPISODE_INTERVAL = CONFIG.SAVE_EPISODE_INTERVAL
    LOAD_MEMORY = args.load_buffer or CONFIG.load_buffer
    ON_POLICY = CONFIG.onpolicy
    SAVE_DATA = args.save_buffer
    SAVE_ONLINE_DATA = args.save_online_buffer
    LOAD_SCENE = args.load_scene
    MERGE_EVERY = 1
    ENV_RESET_TRIALS = CONFIG.ENV_RESET_TRIALS
    LOAD_OBJ_NUM = CONFIG.load_obj_num
    EXTEND_MAX_STEP = MAX_STEP + 6

    DAGGER_MIN_STEP = CONFIG.DAGGER_MIN_STEP
    DAGGER_MAX_STEP = CONFIG.DAGGER_MAX_STEP
    EXPERT_INIT_MIN_STEP = CONFIG.EXPERT_INIT_MIN_STEP
    EXPERT_INIT_MAX_STEP = CONFIG.EXPERT_INIT_MAX_STEP
    DAGGER_RATIO = CONFIG.DAGGER_RATIO
    ENV_RESET_TRIALS = CONFIG.ENV_RESET_TRIALS
    SAVE_EPISODE_INTERVAL = CONFIG.SAVE_EPISODE_INTERVAL

    # cpu and gpu selection
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    CLUSTER = check_ngc()
    MEMORY_THRE = 92
    VISDOM = args.visdom
    actor_wrapper, agent_wrapper, max_memory, NUM_REMOTES = choose_setup()

    # hyperparameters
    object_store_memory = int(max_memory * 1e9)
    reinit_interval = int(LOAD_OBJ_NUM * CONFIG.reinit_factor)
    input_dim = CONFIG.feature_input_dim
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)

    # log
    pretrained_path, logdir, init_pretrained_path = start_log()
    if VISDOM:
        from visdom import Visdom
        vis = Visdom(port=8097 )
        vis.close(None)

    # ray objects
    actors, rollout_agent_id, learner_id, trainer, buffer_id, online_buffer_id = get_ray_objects()
    weights = ray.get(learner_id.get_weight.remote())

    # online training
    os.system('nvidia-smi')
    reinit_count, online_buffer_curr_idx, online_buffer_upper_idx, online_env_step = 0, 0, 0, 0

    for train_iter in itertools.count(1):
        start_rollout = time.time()
        incr_agent_update_step, agent_update_step = ray.get([learner_id.get_agent_incr_update_step.remote(), learner_id.get_agent_update_step.remote()])
        milestone_idx = int((incr_agent_update_step > np.array(CONFIG.mix_milestones)).sum())
        explore_ratio = min(get_valid_index(CONFIG.explore_ratio_list, milestone_idx), CONFIG.explore_cap)
        explore = (np.random.uniform() < explore_ratio) #
        noise_scale = CONFIG.action_noise * get_valid_index(CONFIG.noise_ratio_list, milestone_idx)

        ######################### Rollout and Train
        test_rollout = np.random.uniform() < 0.1
        rollouts = []
        rollouts.extend([actor.rollout.remote(MERGE_EVERY, explore, False, test_rollout, noise_scale) for i, actor in enumerate(actors)])
        rollouts.extend([trainer.train_iter.remote()])
        rollouts.extend([rollout_agent_id_.load_weight.remote(weights) for rollout_agent_id_ in rollout_agent_id])
        rollouts.extend([learner_id.get_weight.remote()])
        res = ray.get(rollouts)
        weights = res[-1]

        ######################### Check Reinit
        buffer_is_full = ray.get(buffer_id.get_info.remote())[2]
        if ON_POLICY: online_buffer_is_full = ray.get(online_buffer_id.get_info.remote())[2]
        buffer_log = get_buffer_log()
        trainer.write_buffer_info.remote(buffer_log)
        trainer.write_external_info.remote( reinit_count=reinit_count, explore_ratio=explore_ratio)
        gpu_max, memory_max, iter_max = log_info()

        if  iter_max:
            reinit()
            reinit_count += 1

        if  memory_max:
            actors, rollout_agent_id, learner_id, trainer, buffer_id, online_buffer_id = reinit(reset=True)

        ######################### Exit
        if (SAVE_DATA and buffer_is_full):
            ray.get(buffer_id.save.remote(cfg.RL_SAVE_DATA_ROOT_DIR))
            break

        if  ON_POLICY and SAVE_ONLINE_DATA and online_buffer_is_full:
            ray.get(online_buffer_id.save.remote(cfg.RL_SAVE_DATA_ROOT_DIR))
            break

        if agent_update_step >= CONFIG.max_epoch:
            break
