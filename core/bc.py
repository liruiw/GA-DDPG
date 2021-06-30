# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *
from core.agent import Agent 

class BC(Agent):
    def __init__(self, num_inputs, action_space, args):
        super(BC, self).__init__(num_inputs, action_space, args, name='BC')

    def load_weight(self, weights):
        self.policy.load_state_dict(weights[0])
        self.goal_feature_extractor.load_state_dict(weights[1])
        self.state_feature_extractor.load_state_dict(weights[2])
      
    def get_weight(self):
        return [
                self.policy.state_dict(),
                self.goal_feature_extractor.state_dict(),
                self.state_feature_extractor.state_dict(),
            ]
    
    def extract_feature(self, image_batch, state_input_batch, action_batch=None, goal_batch=None, time_batch=None, 
                                vis=False, value=False, repeat=False, train=True, curr_joint=None):
        feature = self.unpack_batch(image_batch, 
                                    state_input_batch, 
                                    vis=vis,
                                    gt_goal=goal_batch, 
                                    val=value,
                                    repeat=repeat,
                                    train=train)
 
        if self.use_time: feature = torch.cat((feature, time_batch[:,None]), dim=1)
        return feature 

    def update_parameters(self, batch_data, updates, k):
        self.set_mode(False)
        self.prepare_data(batch_data)
 
        state_batch  =  self.extract_feature(
                            self.image_state_batch,
                            self.point_state_batch,
                            goal_batch=self.goal_batch,
                            time_batch=self.time_batch,
                            train=True)
        self.pi, self.log_pi, _, self.aux_pred = self.policy.sample(state_batch)
        loss = self.compute_loss()

        self.optimize(loss, self.update_step)
        self.update_step += 1
        self.log_stat()
        return {k: float(getattr(self, k)) for k in self.loss_info }