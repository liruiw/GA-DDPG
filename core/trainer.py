# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import ray
from core.utils import *
from core.replay_memory import BaseMemory
import datetime
import os
import IPython
from tensorboardX import SummaryWriter

class ReplayMemoryWrapperBase(object):
    """
    Wrapper class for the replay buffer
    """
    def __init__(self, buffer_size, args, name):
        self.memory = BaseMemory(buffer_size, args, name)

    def sample(self, *args):
        return self.memory.sample(*args)

    def sample_episode(self, *args):
        return self.memory.sample_episode(*args)

    def load(self, *args):
        return self.memory.load(*args)

    def push(self, *args):
        return self.memory.push(*args)

    def get_total_lifted(self, *args):
        return self.memory.get_total_lifted(*args)

    def is_full(self):
        return self.memory.is_full

    def save(self, *args):
        return self.memory.save(*args)

    def add_episode(self, *args):
        return self.memory.add_episode(*args)

    def upper_idx(self):
        return self.memory.upper_idx()

    def reset(self):
        return self.memory.reset()

    def get_cur_idx(self):
        return self.memory.get_cur_idx()

    def reward_info(self, *args):
        return self.memory.reward_info(*args)

    def print_obj_performance(self, *args):
        return self.memory.print_obj_performance(*args)

    def get_total_env_step(self):
        return self.memory.get_total_env_step()

    def get_info(self):
        return (self.memory.upper_idx(), self.memory.get_cur_idx(),  self.memory.is_full,
                self.memory.print_obj_performance(), self.memory.get_total_env_step(),
                self.memory.get_expert_upper_idx(), 0.)

class AgentWrapper(object):
    """
    Wrapper class for agent training and logging
    """
    def __init__(self, args_, config_,  pretrained_path=None, input_dim=512,
                       logdir=None, set_init_step=False, model_surfix='latest', model_path=None, buffer_id=None):

        from core.bc import BC
        from core.ddpg import DDPG
        from core import networks
        self.args = args_
        self.config = config_.RL_TRAIN
        self.cfg = config_ # the global one
        torch.manual_seed(self.args.seed)
        POLICY = self.args.policy

        self.agent = eval(POLICY)(input_dim, PandaTaskSpace6D(), self.config)
        net_dict = make_nets_opts_schedulers(self.cfg.RL_MODEL_SPEC,  self.config)
        self.agent.setup_feature_extractor(net_dict)
        self.buffer_id = buffer_id

        self.model_path = model_path
        self.updates = self.agent.load_model(pretrained_path, set_init_step=set_init_step, surfix=model_surfix)
        self.initial_updates = self.updates
        self.epoch = 0


    def get_agent_update_step(self):
        """ get agent update step """
        return self.agent.update_step

    def get_agent_incr_update_step(self):
        """ get agent update step """
        return self.agent.update_step - self.initial_updates

    def load_weight(self, weights):
        """ get agent update step """
        self.agent.load_weight(weights)
        return [0]

    def get_weight(self):
        """ get agent update step """

        return self.agent.get_weight()

    def save_model(self, surfix='latest'):
        """ save model """
        self.agent.save_model(self.agent.update_step, output_dir=self.config.model_output_dir, surfix=surfix)

    def get_agent(self):
        return self.agent

    def select_action(self, state, actions=None, goal_state=None, remain_timestep=1,
                      gt_goal_rollout=True, curr_joint=None, gt_traj=None):
        """ on policy action """
        action, traj, extra_pred, aux_pred = self.agent.select_action(state, actions=actions, goal_state=goal_state )
        return action, traj, extra_pred, aux_pred

    def update_parameter(self, batch_data, updates, i):
        return self.agent.update_parameters(batch_data,  updates, i)

    def get_agent_lr_info(self):
        self.agent.step_scheduler(self.agent.update_step)
        return (self.agent.update_step, self.agent.get_lr())


class Trainer(object):
    """
    Wrapper class for agent training and logging
    """
    def __init__(self, args_, config_, agent_remote_id, buffer_remote_id, online_buffer_remote_id, logdir=None, model_path=None):
        #
        self.args = args_
        self.config = config_.RL_TRAIN
        self.cfg = config_ # the global one
        torch.manual_seed(self.cfg.RNG_SEED)
        self.agent_remote_id = agent_remote_id
        self.model_path = model_path
        self.buffer_remote_id = buffer_remote_id
        self.online_buffer_remote_id = online_buffer_remote_id
        self.epoch = 0
        self.updates = ray.get(agent_remote_id.get_agent_update_step.remote())
        self.agent_update_step = self.updates
        if logdir is not None:
            self.writer = SummaryWriter(logdir=logdir)
        self.file_handle = None
        self.losses_info = get_loss_info_dict()

    def save_epoch_model(self, update_step):
        ray.get(self.agent_remote_id.save_model.remote(surfix='epoch_{}'.format(update_step)))
        print_and_write(self.file_handle, 'save model path: {} {} step: {}'.format(self.config.output_time, self.config.logdir, update_step))

    def save_latest_model(self, update_step):
        ray.get(self.agent_remote_id.save_model.remote())
        print_and_write(self.file_handle, 'save model path: {} {} step: {}'.format(self.config.model_output_dir, self.config.logdir, update_step))

    def get_agent_update_step(self):
        return ray.get(self.agent_remote_id.get_agent_update_step.remote())

    def write_config(self):
        """ write loaded config to tensorboard  """
        self.writer.add_text('global args', cfg_repr(self.cfg))
        self.writer.add_text('model spec', cfg_repr(yaml.load(open(self.cfg.RL_MODEL_SPEC).read())))
        self.writer.add_text('output time', self.config.output_time)
        self.writer.add_text('script', cfg_repr(yaml.load(open(os.path.join(self.cfg.SCRIPT_FOLDER, self.args.config_file)).read())))

    def write_buffer_info(self, reward_info):
        names = ['reward', 'avg_reward', 'online_reward', 'test_reward', 'avg_collision', 'avg_lifted']
        for i in range(len(names)):
            eb_info, ob_info = reward_info[i]
            if ob_info != 0: self.writer.add_scalar('reward/ob_{}'.format(names[i]), ob_info, self.updates)
            if eb_info != 0: self.writer.add_scalar('reward/eb_{}'.format(names[i]), eb_info, self.updates)

    def write_external_info(self, reinit_count=0, env_step=0, online_env_step=0, buffer_curr_idx=0, online_buffer_curr_idx=0,
                            noise_scale=0, explore_ratio=0, gpu_usage=0, memory_usage=0, buffer_upper_idx=0, sample_time=0, online_buffer_upper_idx=0,
                             ):
        """
        write external information to the tensorboard
        """
        if sample_time > 0: self.writer.add_scalar('info/avg_sample_time', sample_time, self.updates)
        if gpu_usage > 0: self.writer.add_scalar('info/gpu_usage', gpu_usage, self.updates)
        if memory_usage > 0: self.writer.add_scalar('info/memory_usage', memory_usage, self.updates)
        if reinit_count > 0: self.writer.add_scalar('info/epoch', reinit_count, self.updates)
        if online_env_step > 0: self.writer.add_scalar('info/online_env_step', online_env_step, self.updates)
        if env_step > 0: self.writer.add_scalar('info/env_step', env_step, self.updates)
        if explore_ratio > 0: self.writer.add_scalar('info/explore_ratio', explore_ratio, self.updates)
        if noise_scale > 0: self.writer.add_scalar('info/noise_scale', noise_scale, self.updates)

        if self.agent_update_step > 0: self.writer.add_scalar('info/agent_update_step', self.agent_update_step, self.updates)
        if explore_ratio > 0: self.writer.add_scalar('info/explore_ratio', explore_ratio, self.updates)
        if buffer_curr_idx > 0: self.writer.add_scalar('info/buffer_curr_idx', buffer_curr_idx, self.updates)
        if buffer_upper_idx > 0: self.writer.add_scalar('info/buffer_upper_idx', buffer_upper_idx, self.updates)
        if online_buffer_upper_idx > 0: self.writer.add_scalar('info/online_buffer_upper_idx', online_buffer_upper_idx, self.updates)
        if online_buffer_curr_idx > 0: self.writer.add_scalar('info/online_buffer_curr_idx', online_buffer_curr_idx, self.updates)

    def train_iter(self):
        """
        Run inner loop training and update parameters
        """

        LOG_INTERVAL = 3
        self.epoch += 1
        if self.epoch < self.config.fill_data_step:
            return [0] # collect some traj first
        start_time = time.time()
        batch_size = self.config.batch_size

        onpolicy_batch_size = int(batch_size * self.config.online_buffer_ratio)
        batch_data, online_batch_data = ray.get([self.buffer_remote_id.sample.remote(self.config.batch_size), self.online_buffer_remote_id.sample.remote(onpolicy_batch_size)])

        if self.config.ON_POLICY:
            batch_data = {k: np.concatenate((batch_data[k], online_batch_data[k]), axis=0) for k in batch_data.keys() \
                                if type(batch_data[k]) is np.ndarray and k in online_batch_data.keys()}
        if len(batch_data) == 0: return [0]

        for i in range(self.config.updates_per_step):
            batch_data, online_batch_data, main_loss, (update_step, lrs) = ray.get([
                                                        self.buffer_remote_id.sample.remote(self.config.batch_size),
                                                        self.online_buffer_remote_id.sample.remote(onpolicy_batch_size),
                                                        self.agent_remote_id.update_parameter.remote(batch_data, self.updates, i),
                                                        self.agent_remote_id.get_agent_lr_info.remote()
                                                        ])

            if self.config.ON_POLICY:
                batch_data = {k: np.concatenate((batch_data[k], online_batch_data[k]), axis=0) for k in batch_data.keys() \
                                 if type(batch_data[k]) is np.ndarray and k in online_batch_data.keys()}

            infos = merge_two_dicts(lrs, main_loss)
            for k, v in infos.items():
                if k in self.losses_info: self.losses_info[k].append(v)
            self.updates += 1
            self.agent_update_step = update_step

            if self.args.log and self.updates % LOG_INTERVAL == 0:
                for k, v in infos.items():
                    if v == 0:
                        continue
                    elif k.endswith('loss'):
                        self.writer.add_scalar('loss/{}'.format(k), v, self.updates)
                    elif 'ratio' in k or 'gradient' in k or 'lr' in k or 'scale' in k:
                        self.writer.add_scalar('scalar/{}'.format(k), v, self.updates)
                    else:
                        self.writer.add_scalar('info/{}'.format(k), v, self.updates)

            if self.args.save_model and update_step in self.config.save_epoch:
                self.save_epoch_model(update_step)

        if self.args.save_model and self.epoch % 50 == 0:
            self.save_latest_model(update_step)

        # log
        buffer_info, online_buffer_info = ray.get([self.buffer_remote_id.get_info.remote(), self.online_buffer_remote_id.get_info.remote()])
        buffer_upper_idx, buffer_curr_idx, buffer_is_full, obj_performance_str, env_step, buffer_exp_idx, sample_time = buffer_info
        online_buffer_upper_idx, online_buffer_curr_idx, online_buffer_is_full, online_obj_performance_str, online_env_step, _, _ = online_buffer_info
        gpu_usage, memory_usage = get_usage()
        incr_agent_update_step  = ray.get(self.agent_remote_id.get_agent_incr_update_step.remote())
        milestone_idx = int((incr_agent_update_step > np.array(self.config.mix_milestones)).sum())
        explore_ratio = min(get_valid_index(self.config.explore_ratio_list, milestone_idx), self.config.explore_cap)
        noise_scale = self.config.action_noise * get_valid_index(self.config.noise_ratio_list, milestone_idx)

        self.write_external_info(   env_step=env_step,
                                    online_env_step=online_env_step,
                                    buffer_curr_idx=buffer_curr_idx,
                                    buffer_upper_idx=buffer_upper_idx,
                                    online_buffer_curr_idx=online_buffer_curr_idx,
                                    online_buffer_upper_idx=online_buffer_upper_idx,
                                    gpu_usage=gpu_usage,
                                    memory_usage=memory_usage,
                                    explore_ratio=explore_ratio,
                                    noise_scale=noise_scale )

        train_iter_time = (time.time() - start_time)
        self.writer.add_scalar('info/train_time', train_iter_time, self.updates)
        print_and_write(self.file_handle, '==================================== Learn ====================================')
        print_and_write(self.file_handle, 'model: {}  updates: {} lr: {:.5f}  network time: {:.3f} sample time: {:.3f} buffer: {}/{} {}/{}'.format(
                        self.config.output_time, self.updates, infos['policy_lr'], train_iter_time, sample_time * self.config.updates_per_step,
                        buffer_curr_idx, buffer_upper_idx, online_buffer_curr_idx, online_buffer_upper_idx))

        headers = ['loss name', 'loss val']
        data = [(name, np.mean(list(loss)))
                for name, loss in self.losses_info.items() if np.mean(list(loss)) != 0 ]
        print_and_write(self.file_handle, tabulate.tabulate(data, headers, tablefmt='psql'))
        print_and_write(self.file_handle, '== CONFIG: {} == \n== GPU: {} MEMORY: {} TIME: {} PID: {}'.format(
                        self.cfg.script_name, gpu_usage, memory_usage,
                        datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"),
                        os.getppid()))
        return [0]


@ray.remote(num_cpus=3)
class ReplayMemoryWrapper(ReplayMemoryWrapperBase):
    pass

@ray.remote(num_cpus=1,num_gpus=0.1)
class AgentWrapperGPU05(AgentWrapper):
    pass

@ray.remote(num_cpus=2, num_gpus=2)
class AgentWrapperGPU2(AgentWrapper):
    pass

@ray.remote(num_cpus=3, num_gpus=3)
class AgentWrapperGPU3(AgentWrapper):
    pass

@ray.remote(num_cpus=4, num_gpus=4)
class AgentWrapperGPU4(AgentWrapper):
    pass

@ray.remote(num_cpus=1, num_gpus=1)
class AgentWrapperGPU1(AgentWrapper):
    pass

@ray.remote(num_cpus=1, num_gpus=0.5)
class AgentWrapperGPU05(AgentWrapper):
    pass

@ray.remote(num_cpus=1, num_gpus=0.05)
class RolloutAgentWrapperGPU1(AgentWrapper):
    pass

@ray.remote
class TrainerRemote(Trainer):
    pass