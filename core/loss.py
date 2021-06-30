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
import math


def goal_pred_loss(grasp_pred, goal_batch, huber=False):
    """
    PM loss for grasp pose detection
    """
    grasp_pcs = transform_control_points( grasp_pred, grasp_pred.shape[0], device="cuda", rotz=True )
    grasp_pcs_gt = transform_control_points( goal_batch, goal_batch.shape[0], device="cuda", rotz=True )
    return torch.mean(torch.abs(grasp_pcs - grasp_pcs_gt).sum(-1))

def pose_bc_loss(pi, action_batch, mix_policy_ratio=0, huber=False):
    """
    PM loss for behavior clone
    """
    pred_act_pt = control_points_from_rot_and_trans(pi[: :, 3:], pi[: :, :3], device="cuda")
    gt_act_pt = control_points_from_rot_and_trans( action_batch[: :, 3:], action_batch[: :, :3], device="cuda")
    return torch.mean(torch.abs(pred_act_pt - gt_act_pt).sum(-1) )
  