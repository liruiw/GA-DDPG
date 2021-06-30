# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from core import networks
from core.utils import *
import IPython
import time
from core.loss import *

class Agent(object):
    """
    A general agent class
    """

    def __init__(self, num_inputs, action_space, args, name):
        for key, val in args.items():
            setattr(self, key, val)

        self.name = name
        self.device = "cuda"

        self.update_step = 1
        self.init_step = 1
        self.extra_pred_dim = 1
        self.critic_extra_pred_dim = 0
        self.action_dim = action_space.shape[0]
        self.has_critic = self.name != "BC"
        if self.policy_aux:
            self.extra_pred_dim = 7

        if self.critic_aux:
            self.critic_extra_pred_dim = 7

        if self.use_time:
            num_inputs += 1

        action_dim = 0 if self.value_model else action_space.shape[0]
        action_space = action_space if self.use_action_limit else None
        self.action_space = action_space
        self.num_inputs = num_inputs
        self.policy, self.policy_optim, self.policy_scheduler, self.policy_target = get_policy_class('GaussianPolicy', self)
        self.action_dim = action_dim


    def unpack_batch(
        self,
        state,
        point_state=None,
        vis=False,
        gt_goal=None,
        val=False,
        grasp_set=None,
        vis_image=False,
        train=True,
        repeat=False,
    ):
        """
        Extract features from point cloud input
        """
        if type(point_state) is list or type(point_state) is np.ndarray:
            point_state = torch.FloatTensor(point_state).cuda()
        if type(state) is list or type(state) is np.ndarray:
            state = torch.FloatTensor(state).cuda()

        grasp = None
        point_state_feature, network_input = self.state_feature_extractor(
            point_state,
            grasp=grasp,
            concat_option=self.concat_option,
            feature_2=val,
            train=train,
        )

        return point_state_feature

    @torch.no_grad()
    def select_action(
        self,
        state,
        actions=None,
        goal_state=None,
        vis=False,
        remain_timestep=0,
        grasp_set=None,
        gt_goal_rollout=False,
        repeat=False,
    ):

        self.goal_feature_extractor.eval()
        self.state_feature_extractor.eval()
        self.policy.eval()

        if goal_state is not None:
            goal_state = torch.FloatTensor(goal_state).view(1, -1).cuda()

        img_state = torch.FloatTensor(state[0][1][None]).cuda()
        point_state = torch.FloatTensor(state[0][0][None]).cuda()
        timestep = torch.Tensor([remain_timestep]).float().cuda()
        feature = self.extract_feature(
            img_state,
            point_state,
            time_batch=timestep,
            goal_batch=goal_state,
            vis=vis,
            value=False,
            repeat=repeat,
            train=False,
        )

        actions = self.policy.sample(feature)
        action = actions[0].detach().cpu().numpy()[0]
        extra_pred = actions[1].detach().cpu().numpy()[0][0]
        action_sample = actions[2].detach().cpu().numpy()[0]
        aux_pred = actions[3]
        if actions[3] is not None and self.policy_aux:
            aux_pred = aux_pred.detach().cpu().numpy()[0]
        else:
            aux_pred = goal_state.detach().cpu().numpy()[0]
        return action, extra_pred, action_sample, aux_pred

    def compute_loss(self):
        """
        compute loss for policy and trajectory embedding
        """


        if self.policy_aux:
            self.policy_grasp_aux_loss =  goal_pred_loss(self.aux_pred[self.goal_reward_mask, :7], self.target_grasp_batch[self.goal_reward_mask, :7] )

        self.bc_loss =  pose_bc_loss(self.pi[self.expert_mask], self.expert_action_batch[self.expert_mask] )
        if self.has_critic:
            self.bc_loss = self.bc_loss * (1 - self.mix_policy_ratio)
        return sum([getattr(self, name) for name in self.loss_info if name.endswith('loss') and not name.startswith('critic')])


    def update_parameters(self, batch_data, updates, k):
        """
        To be inherited
        """
        return {}


    def setup_feature_extractor(self, net_dict, eval=False):
        """
        Load networks
        """
        state_feature_extractor = net_dict["state_feature_extractor"]
        goal_feature_extractor = net_dict["goal_feature_extractor"]
        self.goal_feature_extractor = goal_feature_extractor["net"]
        self.goal_feature_extractor_opt = goal_feature_extractor["opt"]
        self.goal_feature_extractor_sch = goal_feature_extractor[ "scheduler" ]
        self.state_feature_extractor =state_feature_extractor["net"]
        self.state_feature_extractor_optim = state_feature_extractor["opt"]
        self.state_feature_extractor_scheduler = state_feature_extractor[ "scheduler" ]
        self.state_feat_encoder_optim = state_feature_extractor[  "encoder_opt" ]
        self.state_feat_encoder_scheduler = state_feature_extractor[ "encoder_scheduler" ]
        self.state_feat_val_encoder_optim = state_feature_extractor[ "val_encoder_opt"  ]
        self.state_feat_val_encoder_scheduler =state_feature_extractor[ "val_encoder_scheduler" ]

    def get_lr(self):
        """
        Get network learning rates
        """
        lrs = {
            "policy_lr": self.policy_optim.param_groups[0]["lr"],
            "feature_lr": self.state_feature_extractor_optim.param_groups[0]["lr"],
            "value_lr": self.critic_optim.param_groups[0]["lr"]
            if hasattr(self, "critic")
            else 0,
        }
        return lrs

    def step_scheduler(self, step=None):
        """
        Update network scheduler
        """

        if hasattr(self, "critic"):
            self.critic_scheduler.step()  # step
        if hasattr(self, "policy"):
            self.policy_scheduler.step()  # step
        if self.train_feature or self.train_value_feature:
            self.state_feature_extractor_scheduler.step()  # step
            self.state_feat_encoder_scheduler.step()

    def optimize(self, loss, update_step):
        """
        Backward loss and update optimizer
        """
        self.state_feat_encoder_optim.zero_grad()
        self.policy_optim.zero_grad()
        loss.backward()  #

        self.policy_optim.step()
        self.goal_feature_extractor_opt.step()
        if self.train_feature:
            self.state_feat_encoder_optim.step()
        if hasattr(self, "policy_target"):
            soft_update(self.policy_target, self.policy, self.tau)
        if hasattr(self, "critic_target"):
            half_soft_update(self.critic_target, self.critic, self.tau)
            if update_step % self.target_update_interval == 0:
                half_hard_update(self.critic_target, self.critic, self.tau)

    def prepare_data(self, batch_data):
        """
        load batch data dictionary and compute extra data
        """
        update_step = self.update_step - self.init_step
        self.loss_info  = list(get_loss_info_dict().keys())

        for name in self.loss_info:
            setattr(self, name, torch.zeros(1, device=torch.device('cuda')))

        for k, v in batch_data.items():
            setattr(self, k, torch.cuda.FloatTensor(v))

        self.reward_mask = (self.return_batch > 0).view(-1)
        self.expert_mask = (self.expert_flag_batch >= 1).view(-1)

        self.expert_reward_mask = self.reward_mask * (self.expert_flag_batch >= 1).squeeze()
        self.perturb_flag_batch = (self.perturb_flag_batch < 1).bool()
        self.goal_reward_mask = torch.ones_like(self.time_batch).bool() * self.reward_mask
        self.target_grasp_batch = self.goal_batch[:, :7]
        self.target_goal_reward_mask =  self.goal_reward_mask # target_goal_reward_mask
        self.target_reward_mask =  self.reward_mask
        self.target_return = self.return_batch
        self.target_expert_mask = self.expert_mask
        self.target_expert_reward_mask =  self.expert_reward_mask

        self.target_perturb_flag_batch =  self.perturb_flag_batch < 1
        self.next_time_batch = self.time_batch - 1
        self.target_reward_batch = self.reward_batch
        self.target_mask_batch = self.mask_batch

    def log_stat(self):
        """
        log grad and param statistics for tensorboard
        """
        self.policy_grad = module_max_gradient(self.policy)
        self.feat_grad = module_max_gradient(self.state_feature_extractor.module.encoder)
        self.feat_param = module_max_param(self.state_feature_extractor.module.encoder)
        self.val_feat_grad = module_max_gradient(self.state_feature_extractor.module.value_encoder)
        self.val_feat_param = module_max_param(self.state_feature_extractor.module.value_encoder)
        self.policy_param = module_max_param(self.policy)
        self.reward_mask_num = self.reward_mask.float().sum()

        if self.has_critic:
            self.reward_mask_num = self.reward_mask.sum()
            self.return_min, self.return_max = self.return_batch.min(), self.return_batch.max()

            self.critic_grad = module_max_gradient(self.critic)
            self.critic_param = module_max_param(self.critic)

    def set_mode(self, test):
        """
        set training or test mode for network
        """
        self.test_mode = test

        if not test:
            self.state_feature_extractor.train()
            self.policy.train()

            if hasattr(self, "critic"):
                self.critic.train()
                self.critic_optim.zero_grad()
                self.state_feat_val_encoder_optim.zero_grad()

        else:
            torch.no_grad()
            self.policy.eval()
            self.state_feature_extractor.eval()
            if hasattr(self, "critic"): self.critic.eval()

    def save_model(
        self,
        step,
        output_dir="",
        surfix="latest",
        actor_path=None,
        critic_path=None,
        goal_feat_path=None,
        state_feat_path=None,
    ):
        """
        save model
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if actor_path is None:
            actor_path = "{}/{}_actor_{}_{}".format(
                output_dir, self.name, self.env_name, surfix )
        if critic_path is None:
            critic_path = "{}/{}_critic_{}_{}".format(
                output_dir, self.name, self.env_name, surfix  )
        if goal_feat_path is None:
            goal_feat_path = "{}/{}_goal_feat_{}_{}".format(
                output_dir, self.name, self.env_name, surfix )
        if state_feat_path is None:
            state_feat_path = "{}/{}_state_feat_{}_{}".format(
                output_dir, self.name, self.env_name, surfix )

        print("Saving models to {} and {}".format(actor_path, critic_path))
        if hasattr(self, "policy"):
            torch.save(
                {
                    "net": self.policy.state_dict(),
                    "opt": self.policy_optim.state_dict(),
                    "sch": self.policy_scheduler.state_dict(),
                },
                actor_path,
            )
        if hasattr(self, "critic"):
            torch.save(
                {
                    "net": self.critic.state_dict(),
                    "opt": self.critic_optim.state_dict(),
                    "sch": self.critic_scheduler.state_dict(),
                },
                critic_path,
            )


        if self.use_point_state:
            torch.save(
                {
                    "net": self.state_feature_extractor.state_dict(),
                    "opt": self.state_feature_extractor_optim.state_dict(),
                    "encoder_opt": self.state_feat_encoder_optim.state_dict(),
                    "sch": self.state_feature_extractor_scheduler.state_dict(),
                    "encoder_sch": self.state_feat_encoder_scheduler.state_dict(),
                    "val_encoder_opt": self.state_feat_val_encoder_optim.state_dict(),
                    "val_encoder_sch": self.state_feat_val_encoder_scheduler.state_dict(),
                    "step": step,
                },
                state_feat_path,
            )

    def load_model(
        self, output_dir, surfix="latest", set_init_step=False, reinit_value_feat=False
    ):
        """
        Load saved model
        set_init_step: first run, hack for intermediate restart.
        """
        actor_path = "{}/{}_actor_{}_{}".format(
            output_dir, self.name, self.env_name, surfix
        )
        critic_path = "{}/{}_critic_{}_{}".format(
            output_dir, self.name, self.env_name, surfix
        )
        goal_feat_path = "{}/{}_goal_feat_{}_{}".format(
            output_dir, self.name, self.env_name, surfix
        )
        state_feat_path = "{}/{}_state_feat_{}_{}".format(
            output_dir, self.name, self.env_name, surfix
        )

        if hasattr(self, "policy") and os.path.exists(actor_path):
            net_dict = torch.load(actor_path)
            self.policy.load_state_dict(net_dict["net"])

            self.policy_optim.load_state_dict(net_dict["opt"])
            self.policy_scheduler.load_state_dict(net_dict["sch"])

            if self.reinit_optim and set_init_step:
                for g in self.policy_optim.param_groups:
                    g["lr"] = self.reinit_lr
                self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.policy_optim, milestones=self.policy_milestones, gamma=0.5
                )
                self.policy_scheduler.initial_lr = self.reinit_lr
                self.policy_scheduler.base_lrs[0] = self.reinit_lr
                print("reinit policy optim")

            print("load pretrained policy!!!!")
            hard_update(self.policy_target, self.policy, self.tau)

        if hasattr(self, "critic") and os.path.exists(critic_path):
            net_dict = torch.load(critic_path)
            self.critic.load_state_dict(net_dict["net"])
            self.critic_optim.load_state_dict(net_dict["opt"])
            self.critic_scheduler.load_state_dict(net_dict["sch"])

            if self.reinit_optim and set_init_step:
                for g in self.critic_optim.param_groups:
                    g["lr"] = self.reinit_lr
                self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.critic_optim, milestones=self.value_milestones, gamma=0.5
                )
                self.critic_scheduler.initial_lr = self.reinit_lr
                self.critic_scheduler.base_lrs[0] = self.reinit_lr

            print("load pretrained critic!!!!")
            hard_update(self.critic_target, self.critic, self.tau)


        if os.path.exists(state_feat_path):
            net_dict = torch.load(state_feat_path)
            self.state_feature_extractor.load_state_dict(net_dict["net"])
            try:
                self.state_feature_extractor_optim.load_state_dict(net_dict["opt"])
                self.state_feature_extractor_scheduler.load_state_dict(net_dict["sch"])
                self.state_feat_encoder_optim.load_state_dict(net_dict["encoder_opt"])
                self.state_feat_encoder_scheduler.load_state_dict( net_dict["encoder_sch"])
                self.state_feat_val_encoder_optim.load_state_dict( net_dict["val_encoder_opt"])
                self.state_feat_val_encoder_scheduler.load_state_dict(net_dict["val_encoder_sch"])
                print("load feat optim")
            except:
                print("loading feature optim has mismatches")

            print(
                "load pretrained feature!!!! from: {} step :{}".format(
                    state_feat_path, net_dict["step"]
                )
            )
            self.update_step = net_dict["step"]
            if set_init_step:
                self.init_step = self.update_step

            return self.update_step
        return 0