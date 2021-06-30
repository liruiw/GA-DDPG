# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *
from torch.optim import Adam
from core.agent import Agent
from core import networks
from core.loss import *

class DDPG(Agent):
    def __init__(self, num_inputs, action_space, args):
        super(DDPG, self).__init__(num_inputs, action_space, args, name='DDPG')
        action_dim = 0 if self.value_model else action_space.shape[0]
        self.critic_num_input = num_inputs + 1
        self.critic_value_dim = 0
        self.critic, self.critic_optim, self.critic_scheduler, self.critic_target = get_critic(self)

    def load_weight(self, weights):
        self.policy.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        self.goal_feature_extractor.load_state_dict(weights[2])
        self.state_feature_extractor.load_state_dict(weights[3])

    def get_weight(self):
        return [
            self.policy.state_dict(),
            self.critic.state_dict(),
            self.goal_feature_extractor.state_dict(),
            self.state_feature_extractor.state_dict(),
        ]

    def extract_feature(self, image_batch, point_batch, action_batch=None,
                                            goal_batch=None, time_batch=None,
                                            vis=False, value=False, repeat=False,
                                            train=True, curr_joint=None):
        """
        extract features for policy learning
        """
        point_batch = point_batch.clone()
        if self.sa_channel_concat and value:
            point_batch = concat_state_action_channelwise(point_batch, action_batch)

        value = value and not self.shared_feature #
        feature  =  self.unpack_batch(  image_batch,
                                        point_batch,
                                        vis=vis,
                                        gt_goal=goal_batch,
                                        val=value,
                                        repeat=repeat,
                                        train=train)

        if self.use_time:
            feature = torch.cat((feature, time_batch[:,None]), dim=1)

        return feature

    def target_value(self):
        """
        compute target value
        """
        next_time_batch = self.time_batch - 1
        reward_batch =  self.reward_batch
        mask_batch =   self.mask_batch

        with torch.no_grad():
            next_state_batch  = self.extract_feature(self.next_image_state_batch,
                                                     self.next_point_state_batch,
                                                     action_batch=self.next_action_batch,
                                                     goal_batch=self.next_goal_batch,
                                                     time_batch=next_time_batch,
                                                     value=False)

            next_action_mean, _, _, _ = self.policy_target.sample(next_state_batch)
            idx = int((self.update_step > np.array(self.mix_milestones)).sum())
            noise_scale = self.action_noise * get_valid_index(self.noise_ratio_list, idx)
            noise_delta = get_noise_delta(next_action_mean, noise_scale, self.noise_type)
            noise_delta[:, :3] = torch.clamp(noise_delta[:, :3], -0.01, 0.01)
            next_action_mean = next_action_mean + noise_delta
            next_target_state_batch  = self.extract_feature(self.next_image_state_batch, self.next_point_state_batch, next_action_mean,
                                                            self.next_goal_batch, next_time_batch, value=True)

            min_qf_next_target = self.state_action_value(next_target_state_batch, next_action_mean, target=True)
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target
            return next_q_value


    def state_action_value(self, state_batch, action_batch,  target=False, return_qf=False):
        critic = self.critic_target if target else self.critic
        if self.sa_channel_concat:
            state_batch, action_batch = None, state_batch

        if not self.value_model:
            qf1, qf2, critic_aux = critic(state_batch, action_batch)
        else:
            qf1, qf2, critic_aux = critic(action_batch, None)

        if return_qf:
            return qf1.squeeze(), qf2.squeeze(), critic_aux

        min_qf = torch.min(qf1, qf2)
        min_qf = min_qf.squeeze()
        return min_qf

    def get_mix_ratio(self, update_step):
        """
        Get a mixed schedule for supervised learning and RL
        """
        idx = int((self.update_step > np.array(self.mix_milestones)).sum())
        mix_policy_ratio = get_valid_index(self.mix_policy_ratio_list, idx)
        mix_policy_ratio = min(mix_policy_ratio, self.ddpg_coefficients[4])
        mix_value_ratio  = get_valid_index(self.mix_value_ratio_list, idx)
        mix_value_ratio  = min(mix_value_ratio, self.ddpg_coefficients[3])
        return mix_value_ratio, mix_policy_ratio

    def compute_critic_loss(self, value_feat):
        """
        compute one step bellman error
        """
        self.next_q_value = self.target_value()
        self.qf1, self.qf2, self.critic_grasp_aux = self.state_action_value(value_feat, self.action_batch, return_qf=True)
        self.critic_loss = F.smooth_l1_loss(self.qf1.view(-1)[self.perturb_flag_batch], self.next_q_value[self.perturb_flag_batch]) + \
                           F.smooth_l1_loss(self.qf2.view(-1)[self.perturb_flag_batch], self.next_q_value[self.perturb_flag_batch])
        # F.smooth_l1_loss

        if self.critic_aux:
            self.critic_grasp_aux_loss += goal_pred_loss(self.critic_grasp_aux[self.goal_reward_mask, :7], self.goal_batch[self.goal_reward_mask])

    def critic_optimize(self):
        """
        optimize critic and feature gradient
        """
        self.critic_optim.zero_grad()
        self.state_feat_val_encoder_optim.zero_grad()
        critic_loss = sum([getattr(self, name) for name in get_loss_info_dict().keys() if name.endswith('loss') and name.startswith('critic')])
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.state_feat_val_encoder_optim.step()
        self.critic_optim.step()


    def update_parameters(self, batch_data, updates, k, test=False):
        """ update agent parameters """

        self.mix_value_ratio, self.mix_policy_ratio = self.get_mix_ratio(self.update_step)
        self.set_mode(test)
        self.prepare_data(batch_data)

        # value
        value_feat = self.extract_feature(  self.image_state_batch,
                                            self.point_state_batch,
                                            action_batch=self.action_batch,
                                            goal_batch=self.goal_batch,
                                            time_batch=self.time_batch,
                                            value=True)
        self.compute_critic_loss(value_feat)
        self.critic_optimize()

        # policy
        policy_feat = self.extract_feature( self.image_state_batch,
                                            self.point_state_batch,
                                            goal_batch=self.goal_batch,
                                            time_batch=self.time_batch)

        self.pi, _, _, self.aux_pred = self.policy.sample(policy_feat)
        if self.has_critic and (self.update_step % self.policy_update_gap == 0): # actor critic
            value_pi_feat = self.extract_feature(self.image_state_batch, self.point_state_batch,
                                                 self.pi, self.goal_batch, self.time_batch, value=True)

            self.qf1_pi, self.qf2_pi, critic_aux = self.state_action_value(value_pi_feat, self.pi, return_qf=True)
            self.qf1_pi = self.qf1_pi[~self.expert_reward_mask]
            self.qf2_pi = self.qf2_pi[~self.expert_reward_mask]
            self.actor_critic_loss = -self.mix_policy_ratio * torch.min(self.qf1_pi, self.qf2_pi).mean()

        loss = self.compute_loss()
        self.optimize(loss, self.update_step)
        self.update_step += 1

        # log
        self.log_stat()
        return {k: float(getattr(self, k)) for k in get_loss_info_dict().keys()}