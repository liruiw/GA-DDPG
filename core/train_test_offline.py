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
from tensorboardX import SummaryWriter

from env.panda_scene import PandaYCBEnv, PandaTaskSpace6D, PandaJointSpace
from experiments.config import *
from core.replay_memory import BaseMemory as ReplayMemory
from core import networks
from collections import deque
import glob


from core.utils import *
import json
import scipy.io as sio
import IPython
import pprint
import cv2

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
parser.add_argument('--batch_size',  type=int, default=-1)
parser.add_argument('--fix_output_time', type=str, default=None)


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

    if new_output_dir:
        os.makedirs(model_output_dir)
        script_file = os.path.join(cfg.SCRIPT_FOLDER, args.config_file)
        cfg_from_file(script_file)
        cfg.script_name = args.config_file
        os.system(
            "cp {} {}".format(
                script_file, os.path.join(model_output_dir, args.config_file) ) )
        os.system(
            "cp {} {}".format(
                cfg.RL_MODEL_SPEC,
                os.path.join(model_output_dir, cfg.RL_MODEL_SPEC.split("/")[-1]) ) )
        save_cfg_to_file(os.path.join(model_output_dir, "config.yaml"), cfg)

        if load_from_pretrain:
            migrate_model(args.pretrained, model_output_dir, args.model_surfix)
            print("migrate policy...")

    print("Using config:")
    pprint.pprint(cfg)
    net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    return net_dict, dt_string


def train_off_policy():
    """
    train the network with off-policy saved data
    """
    losses = get_loss_info_dict()

    for epoch in itertools.count(1):
        start_time = time.time()
        lrs = agent.get_lr()
        data_time, network_time = 0., 0.
        for i in range(CONFIG.updates_per_step):
            batch_data = memory.sample(batch_size=CONFIG.batch_size)
            data_time = data_time + (time.time() - start_time)
            start_time = time.time()
            loss = agent.update_parameters(batch_data, agent.update_step, i)

            network_time += (time.time() - start_time)
            for k, v in loss.items():
                if k in losses: losses[k].append(v)

            agent.step_scheduler(agent.update_step)
            start_time = time.time()

            if args.save_model and epoch % 100 == 0 and i == 0:
                agent.save_model(agent.update_step, output_dir=model_output_dir)
                print('save model path: {} {} step: {}'.format(output_time, logdir, agent.update_step))

            if args.save_model and agent.update_step in CONFIG.save_epoch:
                agent.save_model(agent.update_step, output_dir=model_output_dir, surfix='epoch_{}'.format(agent.update_step))
                print('save model path: {} {} step: {}'.format(model_output_dir, logdir, agent.update_step))

            if args.log and agent.update_step % LOG_INTERVAL <= 1:
                for k, v in loss.items():
                    if v == 0: continue
                    if 'loss' in k:
                        writer.add_scalar('loss/{}'.format(k), v, agent.update_step)
                    elif 'ratio' in k or 'gradient' in k or 'lr' in k:
                        writer.add_scalar('scalar/{}'.format(k), v, agent.update_step)
                    elif v != 0:
                        writer.add_scalar('info/{}'.format(k), v, agent.update_step)

        print('==================================== Learn ====================================')
        print('model: {} epoch: {} updates: {} lr: {:.6f} network time: {:.2f}  data time: {:.2f} batch size: {}'.format(
                output_time, epoch, agent.update_step,  lrs['policy_lr'], network_time, data_time, CONFIG.batch_size))

        headers = ['loss name', 'loss val']
        data = [
                (name, np.mean(list(loss)))
                for name, loss in losses.items() if np.mean(list(loss)) != 0
                ]
        print(tabulate.tabulate(data, headers, tablefmt='psql'))
        print('===================================== {} ========================================='.format(cfg.script_name))

        if agent.update_step >= CONFIG.max_epoch:
            break



def test(run_iter=0):
    """
    test agent performance on test scenes
    """
    global cnt, object_performance
    episodes = args.test_episode_num
    k = 0

    if run_iter == 0:
        mkdir_if_missing('output_misc/rl_output_video_{}_{}'.format(video_prefix, POLICY))

    while (k < episodes):

        # sample scene
        start_time = time.time()
        traj, res_obs = [], []
        scene_file = 'scene_{}'.format(int(k))
        data_root = cfg.RL_TEST_SCENE
        scene_indexes.append(scene_file.split('/')[-1])
        state = env.reset(save=False,   scene_file=scene_file ,
                                        data_root_dir=data_root, reset_free=True,
                                        cam_random=0)

        cur_ef_pose = env._get_ef_pose(mat=True)
        cam_intr = get_info(state, 'intr')
        k += 1

        # check scene
        if not check_scene(env, state, cur_ef_pose[:3, :3],
                            object_performance, scene_file,
                            CONFIG.init_distance_low, CONFIG.init_distance_high,  run_iter):
            continue

        # expert
        if CONFIG.use_expert_plan:
            expert_plan, omg_cost, exp_success = env.expert_plan(False, return_success=True)
        cnt = cnt + 1
        max_steps = cfg.RL_MAX_STEP
        expert_traj = None
        init_info = env._get_init_info()
        episode_reward = 0
        episode_steps = 0

        if CONFIG.use_expert_plan and args.expert: # run expert rollout
            expert_traj = []
            for joint_action in expert_plan:
                goal_state = env._get_relative_goal_pose(mat=True)
                action = env.convert_action_from_joint_to_cartesian(joint_action)
                next_state, reward, done, _ = env.step(action, delta=not DELTA_JOINT)
                vis_img = get_info(next_state, 'img', cfg.RL_IMG_SIZE)
                vis_img = draw_grasp_img(vis_img, goal_state, cam_intr, camera_hand_offset_pose, (0, 255, 0))
                expert_traj.append(vis_img)

            expert_episode_reward, res_obs = env.retract(record=True)
            res_obs = [get_info(r, 'img', cfg.RL_IMG_SIZE) for r in res_obs]
            expert_traj.extend(res_obs)
            state = env.reset(save=False, scene_file=scene_file, init_joints=init_joints,
                                        data_root_dir=cfg.RL_TEST_SCENE, reset_free=True,
                                        cam_random=0, enforce_face_target=True)

        # agent rollout
        done = False
        while not done:

            # agent action
            remain_timestep = max(max_steps-episode_steps, 1)
            vis = False
            action, _, _, aux_pred = agent.select_action(state, vis=False, remain_timestep=remain_timestep )

            # visualize
            vis_img = get_info(state, 'img', cfg.RL_IMG_SIZE)
            if goal_involved:
                pred_grasp = unpack_pose_rot_first(aux_pred).dot(rotZ(np.pi/2))
                best_grasp = pred_grasp
                vis_img  = draw_grasp_img(vis_img, best_grasp.dot(rotZ(np.pi/2)), cam_intr, camera_hand_offset_pose)   #

            # step
            next_state, reward, done, env_info = env.step(action, delta=True, vis=False)
            traj.append(vis_img)
            print('step: {} action: {:.3f} rew: {:.2f} '.format(
            episode_steps, np.abs(action[:3]).sum(), reward ))

            # retract
            if (episode_steps == TOTAL_MAX_STEP or done):
                reward, res_obs = env.retract(record=True)
                res_obs = [get_info(r, 'img', cfg.RL_IMG_SIZE) for r in res_obs]
                done = True
                traj.extend(res_obs)

            state = next_state
            episode_reward += reward
            episode_steps += 1

        # log
        lifted = (reward > 0.5)
        avg_reward.update(episode_reward)
        avg_lifted.update(lifted)
        traj_lengths.append(episode_steps)
        if env.target_name not in object_performance:
            object_performance[env.target_name] = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
        object_performance[env.target_name][0].update(lifted)

        if args.record and len(traj) > 5:
            write_video(traj, scene_indexes[-1], expert_traj, cnt % MAX_VIDEO_NUM, cfg.RL_IMG_SIZE, cfg.OUTPUT_MISC_DIR,
                        logdir, env.target_name, '{}_{}'.format(video_prefix, POLICY), False, lifted, False)

        print('=======================================================================')
        print('test: {} max steps: {}, episode steps: {}, return: {:.3f} time {:.3f} avg return: {:.3f}/{:.3f}/{:.3f} model: {} {} dataset: {}'.format(cnt, TOTAL_MAX_STEP,
                            episode_steps, episode_reward, time.time() - start_time, avg_reward.avg, avg_lifted.avg, exp_lifted.avg, args.pretrained, cfg.script_name, CONFIG.index_file))
        print('=======================================================================')
        print('testing script:', args.output_file)

    # write result
    if run_iter == NUM_RUNS - 1 and args.log:
        dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        output_stat_file = os.path.join(cfg.OUTPUT_MISC_DIR, 'rl_output_stat', args.output_file)
        mkdir_if_missing(os.path.join(cfg.OUTPUT_MISC_DIR, 'rl_output_stat'))
        file_handle = open(output_stat_file, 'a+')
        output_text = ''
        output_text += print_and_write(file_handle, '\n')
        output_text += print_and_write(file_handle, "------------------------------------------------------------------")
        output_text += print_and_write(file_handle, 'Test Time: {} Data Root: {}/{} Model: {}'.format(dt_string, cfg.RL_DATA_ROOT_DIR, cfg.RL_SAVE_DATA_NAME, output_time))
        output_text += print_and_write(file_handle, 'Script: {} Index: {}'.format(cfg.script_name, CONFIG.index_file))
        output_text += print_and_write(file_handle, 'Num of Objs: {} Num of Runs: {} '.format(len(object_performance), NUM_RUNS ))
        output_text += print_and_write(file_handle, 'Policy: {} Model Path: {} Step: {}'.format(POLICY,
                        args.pretrained, agent.update_step ))
        output_text += print_and_write(file_handle, "Test Episodes: {} Avg. Length: {:.3f} Index: {}-{} ".format(
                        cnt, np.mean(traj_lengths), scene_indexes[0], scene_indexes[-1] ))
        output_text += print_and_write(file_handle, 'Avg. Performance: (Return: {:.3f} +- {:.5f}) (Success: {:.3f} +- {:.5f})'.format(
                                        avg_reward.avg, avg_reward.std(), avg_lifted.avg, avg_lifted.std()))
        headers = ['object name', 'count', 'success']
        object_performance = sorted(object_performance.items())
        data = [
                (name, info[0].count, int(info[0].sum))
                for name, info in object_performance
                ]
        obj_performance_str = tabulate.tabulate(data, headers, tablefmt='psql')
        output_text += print_and_write(file_handle, obj_performance_str)
        print('testing script:', args.output_file)

if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    net_dict, output_time = setup()
    CONFIG = cfg.RL_TRAIN
    cfg.RL_TEST_SCENE = 'data/gaddpg_scenes'

    # Args
    RENDER = args.render
    TRAIN = not args.test
    MAX_STEP = cfg.RL_MAX_STEP
    TOTAL_MAX_STEP = MAX_STEP * 2
    LOAD_MEMORY = True
    MAX_TEST_PER_OBJ = args.max_cnt_per_obj
    NUM_RUNS = args.num_runs
    MAX_VIDEO_NUM = 50
    LOG_INTERVAL = 4
    CONFIG.output_time = output_time
    CONFIG.off_policy = True
    CONFIG.index_file = 'ycb_large.json'
    POLICY = 'DDPG' if CONFIG.RL else 'BC'
    cnt = 0.

    # Metrics
    input_dim = CONFIG.feature_input_dim
    avg_reward, avg_lifted, exp_lifted = AverageMeter(), AverageMeter(), AverageMeter()
    object_performance = {}
    traj_lengths, scene_indexes = [], []
    video_prefix = 'YCB'
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)
    pretrained_path = model_output_dir

    if hasattr(cfg, 'script_name') and len(cfg.script_name) > 0:
        args.output_file = args.output_file.replace('txt', 'script_{}.txt'.format(cfg.script_name))
        video_prefix = video_prefix + '_' + cfg.script_name
        print('video output: {} stat output: {}'.format(video_prefix, args.output_file))

    # Agent
    action_space = PandaTaskSpace6D()
    agent = globals()[POLICY](input_dim, action_space, CONFIG) # 138
    agent.setup_feature_extractor(net_dict, args.test)
    agent.load_model(pretrained_path, surfix=args.model_surfix, set_init_step=True)
    CONFIG.batch_size = cfg.OFFLINE_BATCH_SIZE
    cfg.ONPOLICY_MEMORY_SIZE = cfg.OFFLINE_RL_MEMORY_SIZE
    cfg.RL_MEMORY_SIZE = cfg.OFFLINE_RL_MEMORY_SIZE

    # Memory
    if LOAD_MEMORY and TRAIN:
        memory = ReplayMemory(int(cfg.RL_MEMORY_SIZE)+1, cfg)
        memory.load(cfg.RL_SAVE_DATA_ROOT_DIR, cfg.RL_MEMORY_SIZE)

    # Environment
    env_config = cfg.env_config
    env_config['renders'] = RENDER
    env_config['random_target'] = False
    env_config['egl_render'] = False
    env_config['domain_randomization'] = False

    # Tensorboard
    logdir = '{}/{}/{}_{}'.format(cfg.OUTPUT_DIR, output_time, CONFIG.env_name, POLICY)
    print('output_time: {} logdir: {}'.format(output_time, logdir))
    scene_prefix =  '{}_scene'.format(CONFIG.index_file)
    MAX_ONLIND_SCENE_NUM = len(glob.glob(os.path.join(cfg.RL_TEST_SCENE, scene_prefix) + '*'))
    file = os.path.join(cfg.EXPERIMENT_OBJ_INDEX_DIR, 'ycb_large.json')
    with open(file) as f: file_dir = json.load(f)
    file_dir = file_dir['test'][:args.test_episode_num ]
    file_dir = [f[:-5].split('.')[0][:-2] for f in file_dir]
    test_file_dir = list(set(file_dir))
    goal_involved = CONFIG.policy_goal or CONFIG.policy_aux  or CONFIG.critic_aux

    # Main
    if TRAIN:
        writer = SummaryWriter(logdir=logdir)
        train_off_policy()
    else:
        for run_iter in range(NUM_RUNS):
            env = eval(CONFIG.env_name)(**env_config)
            env._load_index_objs(test_file_dir)
            state = env.reset(  save=False, data_root_dir=cfg.DATA_ROOT_DIR,  enforce_face_target=True)
            camera_hand_offset_pose = se3_inverse(env.cam_offset)
            test(run_iter=run_iter)
            avg_lifted.set_mean()
            avg_reward.set_mean()
