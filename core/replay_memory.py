# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
from torch.utils.data import Dataset
import IPython
import time
import cv2
import random
 
from core.utils import *
from collections import deque
 

class BaseMemory(Dataset):
    """Defines a generic experience replay memory module."""

    def __init__(self, buffer_size, args, name="expert"):

        self.cur_idx = 0
        self.total_env_step = 0
        self.is_full = False
        self.name = name

        for key, val in args.RL_TRAIN.items():
            setattr(self, key, val)

        self.buffer_size = buffer_size
        self.episode_max_len = args.RL_MAX_STEP
        self.save_data_name = args.RL_SAVE_DATA_NAME
        self.attr_names = [
            "action",
            "pose",
            "point_state",
            "target_idx",
            "reward",
            "terminal",
            "timestep",
            "returns",
            "state_pose",
            "image_state",
            "collide",
            "grasp",
            "perturb_flags",
            "goal",
            "expert_flags",
            "expert_action"
        ]

        for attr in self.attr_names:
            setattr(self, attr, None)
        (
            self._REW,
            self._ONLINE_REW,
            self._TEST_REW,
            self._TOTAL_REW,
            self._TOTAL_CNT,
        ) = (
            deque([0] * 50, maxlen=200),
            deque([0] * 50, maxlen=50),
            deque([0] * 50, maxlen=50),
            0,
            1,
        )
        self.dir = args.RL_DATA_ROOT_DIR
        self.load_obj_performance()
        self.init_buffer()

    def update_reward(self, reward, test, explore, target_name):
        self._TOTAL_REW += reward
        self._TOTAL_CNT += 1
        self._REW.append(reward)
        if explore:
            self._TEST_REW.append(reward) if test else self._ONLINE_REW.append(reward)
        if target_name != "noexists" and target_name not in self.object_performance:
            self.object_performance[target_name] = [0, 0, 0]
        self.object_performance[target_name][0] += 1
        self.object_performance[target_name][1] += reward

    def load_obj_performance(self):
        self.object_performance = {}

    def reward_info(self):
        return (
            self._TOTAL_REW / self._TOTAL_CNT,
            np.mean(list(self._REW)),
            np.mean(list(self._ONLINE_REW)),
            np.mean(list(self._TEST_REW)),
            0.,
            0.
        )

    def print_obj_performance(self):
        s = "===========================performance cnt ========================\n"
        headers = ["object name", "count", "success"]
        object_performance = sorted(self.object_performance.items())
        data = [(name, info[0], info[1]) for name, info in object_performance]
        obj_performance_str = tabulate.tabulate(data, headers, tablefmt="psql")
        e = "====================================================================\n"
        obj_performance_str = s + obj_performance_str + e
        print("interacted objects:", len(self.object_performance))
        return obj_performance_str

    def __len__(self):
        return self.upper_idx()

    def __getitem__(self, idx):

        data = {
            "image_state_batch":  process_image_output(self.image_state[idx]),
            "expert_action_batch": np.float32(self.expert_action[idx]),
            "action_batch": np.float32(self.action[idx]),
            "reward_batch": np.float32(self.reward[idx]),
            "return_batch": np.float32(self.returns[idx]),
            "next_image_state_batch": None,
            "mask_batch": np.float32(self.terminal[idx]),
            "time_batch": np.float32(self.timestep[idx]),
            "point_state_batch": None,
            "next_point_state_batch": None,
            "state_pose_batch": np.float32(self.state_pose[idx]),
            "collide_batch": np.float32(self.collide[idx]),
            "grasp_batch": np.float32(self.grasp[idx]),
            "goal_batch": np.float32(self.goal[idx]),
        }
        return data

    def upper_idx(self):
        return max(self.cur_idx, 1) if not self.is_full else len(self.point_state)

    def is_full(self):
        return self.is_full

    def get_cur_idx(self):
        return self.cur_idx

    def get_expert_upper_idx(self):
        upper_idx = self.upper_idx()
        if self.expert_flags is not None and np.sum(self.expert_flags[:upper_idx]) > 0:
            return np.where(self.expert_flags[:upper_idx] >= 1)[0][-1]
        else:
            return 0

    def get_total_env_step(self):
        return self.total_env_step

    def reset(self):
        self.cur_idx = 0
        self.is_full = False
 
    def recompute_return_with_gamma(self):
        end_indexes  = np.sort(np.unique(self.episode_map))
        copy_returns = self.returns.copy()
       
        for idx in range(len(end_indexes) - 1):
            start = end_indexes[idx]
            end = end_indexes[idx+1] 
            cost_to_go = 0
            for i in range(end - start):
                cur_idx = end + 1
                copy_returns[cur_idx-i-1 ] = self.reward[cur_idx-i-1] + self.gamma ** i * cost_to_go
                cost_to_go = copy_returns[cur_idx-i-1 ]
        self.returns = copy_returns
 
    def sample(self, batch_size):
        """Samples a batch of experience from the buffer."""

        upper_idx = self.upper_idx()
        batch_idx = np.random.randint(self.episode_max_len, upper_idx, batch_size)

        np.random.shuffle(batch_idx)
        data = self[batch_idx]
        self.post_process_batch(data, batch_idx)

        return data

    def push(self, step_dict):
        """
        Push a single data item to the replay buffer
        """        
        if self.action is None:
            self.init_buffer()
        store_idx = self.cur_idx % len(self.point_state)
        if (
            step_dict["point_state"].shape[1] < 100
            or step_dict["point_state"].sum() == 0 
        ):
            return

        attr_names = self.attr_names[:]
        for name in attr_names:
            if name == "image_state":
                if self.use_image:
                    getattr(self, name)[store_idx] =  process_image_input(
                        step_dict[name].copy()
                    )
            elif name in step_dict:
                getattr(self, name)[store_idx] = step_dict[name]

        if self.cur_idx >= len(self.episode_map) - 1:
            self.is_full = True
        self.cur_idx = self.cur_idx + 1
        self.total_env_step += 1

        if self.cur_idx >= len(self.point_state) or self.cur_idx < self.buffer_start_idx:
            self.cur_idx = self.buffer_start_idx
 
    def add_episode(self, episode,  explore=False, test=False):
        """
        Add an rollout to the dataset
        """
        episode_length = len(episode)
        if (not self.RL) and episode[-1]["reward"] < 0.5 and not explore:
            return

        if episode_length > 0:
            self.update_reward(episode[-1]["reward"] > 0.5, test, explore, episode[-1]["target_name"] )
 
        for transition in episode:
            self.push(transition)

        if self.cur_idx - episode_length >= 0 and episode_length > 0:
            cost_to_go = 0
            for i in range(episode_length):
                self.returns[self.cur_idx - i - 1] = (
                    self.reward[self.cur_idx - i - 1] + self.gamma ** i * cost_to_go )
                cost_to_go = self.returns[self.cur_idx - 1 - i]

            self.episode_map[self.cur_idx - episode_length : self.cur_idx] = self.cur_idx - 1
            

    def set_onpolicy_goal(self, data, batch_idx, vis=False ):
        """
        Rewriting the on-policy goals 
        """   
        mask = self.expert_flags[batch_idx] == 0.0
        episode_end = self.episode_map[batch_idx]   
        increment_idx = np.minimum(episode_end, batch_idx + 1).astype(np.int)
        goal_poses = [((se3_inverse(self.state_pose[batch_idx[i]]).dot(  #
                        self.state_pose[episode_end[i]])))  #
                        for i in range(len(batch_idx)) ]
        next_goal_poses = [((se3_inverse(self.state_pose[increment_idx[i]]).dot(  #
                            self.state_pose[episode_end[i]])))  #
                            for i in range(len(batch_idx)) ]
        next_goal_batch = np.array([pack_pose_rot_first(p) for p in next_goal_poses])
        goal_batch = np.array([pack_pose_rot_first(p) for p in goal_poses])
        data["goal_batch"][mask] = goal_batch[mask]
        data["next_goal_batch"][mask] = next_goal_batch[mask]
 
    def post_process_batch(self, data, batch_idx):  # self.process_sparse_sample
        """
        Set some data in batch
        """
        increment_idx = np.minimum(self.episode_map[batch_idx], batch_idx + 1).astype(np.int)

        data["grasp_sample_batch"] = np.zeros([0, 4, 4])  
        data["next_image_state_batch"] =  process_image_output(self.image_state[increment_idx])
        data["next_goal_batch"] = np.float32( self.goal[increment_idx] )  
        data["next_expert_action_batch"] = np.float32(self.expert_action[increment_idx])
        data["next_action_batch"] = np.float32( self.action[increment_idx] )
        data["next_point_state_batch"] = self.point_state[increment_idx]
        data["next_return_batch"] = self.returns[increment_idx]
        data["point_state_batch"] = self.point_state[batch_idx]
        # remaining time step
        data["time_batch"] = np.float32(self.timestep[self.episode_map[batch_idx]]) + 1 - data["time_batch"]
        data["expert_flag_batch"] = np.float32(self.expert_flags[batch_idx])
        data["perturb_flag_batch"] = np.float32(self.perturb_flags[batch_idx])
        data["batch_idx"] = np.uint8(batch_idx)

        if self.self_supervision and self.name != "expert":
            self.set_onpolicy_goal(data, batch_idx)
 
    def load(self, data_dir, buffer_size=100000, **kwargs):
        """
        Load data saved offline
        """
        print(
            "======================= loading memory =======================".format(
                self.cur_idx
            )
        )
        start_time = time.time()
        if not os.path.exists(data_dir):
            return

        if os.path.exists(os.path.join(data_dir, self.save_data_name)):

            data = np.load(
                os.path.join(data_dir, self.save_data_name),
                allow_pickle=True,
                mmap_mode="r",
            )
            for name in self.attr_names + [
                "episode_map",
                "target_idx",
            ]:
                s = time.time()
                print("loading {} ...".format(name))
                data_max_idx = np.amax(data["episode_map"])
                data_name = name
                if (not self.use_image and name == "image_state"):
                    continue
                if name not in data:
                    if name == data_name:
                        print("not in data:", name)
                        continue
                     
                if type(data[data_name]) is not np.ndarray:        
                    setattr(self, name, data[data_name])
                    print(name, getattr(self, name))
                else:
                    getattr(self, name)[:data_max_idx] = data[data_name][:data_max_idx]
                    print(name + " shape:", getattr(self, name).shape)
                print("load {} time: {:.3f}".format(name, time.time() - s))
            
            self.cur_idx = np.amax(data["episode_map"])   
            self.total_env_step = int(data["total_env_step"])
            self.is_full = (  bool(data["is_full"]) and self.cur_idx >= self.buffer_size - 1 )
            self.cur_idx = self.upper_idx()
            self.recompute_return_with_gamma()

            expert_upper_idx = self.get_expert_upper_idx()
            pos = np.where(self.returns[: self.cur_idx] > 0.0)[0][1:-1]
            expert_flag = self.expert_flags[: self.cur_idx] == 1
            print(
                "======================= loaded idx: {} env step: {} success: {:.3f} expert ratio: {:.3f} start idx: {} expert end idx {} is_full: {}=======================".format(
                    self.upper_idx(),
                    self.total_env_step,
                    float(len(pos)) / (self.cur_idx + 1),
                    float(expert_flag.sum()) / (self.cur_idx + 1),
                    self.buffer_start_idx,
                    expert_upper_idx,
                    self.is_full,
                )
            )

    def save(self, save_dir="."):
        """Saves the current buffer to memory."""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        s = time.time()
        save_dict = {}
        save_attrs = self.attr_names + [
            "episode_map",
            "is_full",
            "cur_idx",
            "total_env_step",
            "target_idx",
        ]
        for name in save_attrs:
            save_dict[name] = getattr(self, name)
        np.savez(os.path.join(save_dir, self.save_data_name), **save_dict)
        print("Saving buffer at {}, time: {:.3f}".format(save_dir, time.time() - s))


    def init_buffer(self):
        if not self.use_image:
            state_size = (1,)  # dummy
        else:
            state_size = (5, 112, 112)
        action_size = (6,)
        pose_size = (64,)

        self.image_state = np.zeros((self.buffer_size,) + state_size, dtype=np.uint16)
        self.action = np.zeros((self.buffer_size,) + action_size, dtype=np.float32)
        self.expert_action = np.zeros((self.buffer_size,) + action_size, dtype=np.float32)

        self.terminal = np.zeros((self.buffer_size,), dtype=np.float32)
        self.timestep = np.zeros((self.buffer_size,), dtype=np.float32)
        self.reward = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pose = np.zeros((self.buffer_size,) + pose_size, dtype=np.float32)
        self.point_state = np.zeros([self.buffer_size, 4, self.uniform_num_pts + 6])  
        self.collide = np.zeros((self.buffer_size,), dtype=np.float32)
        self.grasp = np.zeros((self.buffer_size,), dtype=np.float32)
        self.state_pose = np.zeros((self.buffer_size, 4, 4), dtype=np.float32)
        self.target_idx = np.zeros((self.buffer_size,), dtype=np.float32)
        self.goal = np.zeros((self.buffer_size, 7), dtype=np.float32)
        self.episode_map = np.zeros((self.buffer_size,), dtype=np.uint32)
        self.expert_flags = np.zeros((self.buffer_size,), dtype=np.float32)
        self.perturb_flags = np.zeros((self.buffer_size,), dtype=np.float32)
  