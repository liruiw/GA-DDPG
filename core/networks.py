# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn
import IPython
import numpy as np
import sys
import pointnet2_ops.pointnet2_modules as pointnet2
from core.utils import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def _resnet(arch, block, layers, pretrained, input_dim, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if input_dim > 3:
        model.conv1 = nn.Conv2d(
            input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        model_dict = model.state_dict()
        if model_dict["conv1.weight"].shape[1] > 3:
            print("extend the original conv1 to support additional input")
            old_conv1_weight = pretrained_dict["conv1.weight"]
            ext_conv1_weight = torch.zeros_like(model_dict["conv1.weight"])
            ext_conv1_weight[:, :3, :, :] = old_conv1_weight
            pretrained_dict["conv1.weight"] = ext_conv1_weight

        model.load_state_dict(pretrained_dict, strict=False)  # reset fc weight maybe
    return model


def resnet18(pretrained=False, progress=True, input_dim=3, **kwargs):
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, input_dim, progress, **kwargs
    )


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 64 * scale, 64 * scale, 128 * scale],
    )
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale],
    )

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale]
    )

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(
        nn.Linear(int(512 * scale), int(1024 * scale)),
        nn.BatchNorm1d(int(1024 * scale)),
        nn.ReLU(True),
        nn.Linear(int(1024 * scale), int(512 * scale)),
        nn.BatchNorm1d(int(512 * scale)),
        nn.ReLU(True),
    )
    return nn.ModuleList([sa_modules, fc_layer])


class Identity(nn.Module):
    def forward(self, input):
        return input


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ResNetFeature(nn.Module):
    def __init__(
        self,
        unfreeze_last_k_layer=0,
        pretrained=True,
        input_dim=3,
        batch_norm=True,
        action_concat=False,
        policy_extra_latent=-1,
        critic_extra_latent=-1,
    ):
        super(ResNetFeature, self).__init__()
        self.encoder = resnet18(pretrained=pretrained, input_dim=input_dim)
        self.encoder.train(not pretrained)
        self.encoder.fc = Identity()
        self.batch_norm = batch_norm
        self.q = nn.Linear(512, 4)  # for graspnet
        self.t = nn.Linear(512, 3)
        self.confidence = nn.Linear(512, 1)

        self.value_encoder = resnet18(pretrained=pretrained, input_dim=input_dim)
        self.value_encoder.train(not pretrained)
        self.value_encoder.fc = Identity()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406, 0, 0], dtype=torch.float).reshape((1, 5, 1, 1)))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225, 1, 1], dtype=torch.float).reshape((1, 5, 1, 1)))

    def forward(self, input, val=False, goal_head=False):
        batch_size, c, h, w = input.size()
        input = (input - self.mean.data[:, :c]) / self.std.data[:, :c]
        if val:
            return self.value_encoder(input)
        return self.encoder(input)

        z = self(*args)
        qt = torch.cat((F.normalize(self.q(z), p=2, dim=-1), self.t(z)), -1)
        confidence = self.confidence(z)

    def grasp_pred(self, *args):
        sample_grasps, _ = self(*args, goal_head=True)
        return sample_grasps


class GoalFeature(nn.Module):
    def __init__(
        self,
        input_dim=3,
        pointnet_radius=0.02,
        pointnet_nclusters=128,
        model_scale=1,
        action_concat=False
    ):
        super(GoalFeature, self).__init__()
        self.num_grasp_samples = 1
        self.encoder = base_network(
            pointnet_radius, pointnet_nclusters, model_scale, 3
        )
        self.q = nn.Linear(model_scale * 512, 4)
        self.t = nn.Linear(model_scale * 512, 3)
        self.confidence = nn.Linear(model_scale * 512, 1)

    def encode(self, xyz, xyz_features):
        for module in self.encoder[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return self.encoder[1](xyz_features.squeeze(-1))

    def forward(self, pc, grasp=None, goal_head=False):
        pc = pc.cuda()
        z = self.encode(pc[..., :3], pc.transpose(1, -1).contiguous())
        qt = torch.cat((F.normalize(self.q(z), p=2, dim=-1), self.t(z)), -1)
        confidence = self.confidence(z)
        return qt, torch.sigmoid(confidence).squeeze()



class PointNetFeature(nn.Module):
    def __init__(
        self,
        input_dim=3,
        pointnet_nclusters=32,
        pointnet_radius=0.02,
        model_scale=1,
        extra_latent=0,
        split_feature=False,
        policy_extra_latent=-1,
        critic_extra_latent=-1,
        action_concat=False,
    ):
        super(PointNetFeature, self).__init__()
        self.input_dim = 3 + extra_latent
        self.split_feature = False
        input_dim =  3 + policy_extra_latent if policy_extra_latent > 0 else self.input_dim

        self.policy_input_dim = input_dim
        self.encoder = self.create_encoder(
            model_scale, pointnet_radius, pointnet_nclusters, self.policy_input_dim
        )
        input_dim = 3 + critic_extra_latent if critic_extra_latent > 0 else input_dim
        self.critic_input_dim = input_dim
        if action_concat:
            self.critic_input_dim = 10
        self.value_encoder = self.create_encoder(
            model_scale, pointnet_radius, pointnet_nclusters, self.critic_input_dim
        )

    def create_encoder(
        self, model_scale, pointnet_radius, pointnet_nclusters, input_dim=0
    ):
        return base_network(pointnet_radius, pointnet_nclusters, model_scale, input_dim)

    def encode(self, encoder, xyz, xyz_features):
        for module in encoder[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return encoder[1](xyz_features.squeeze(-1))

    def forward(
        self,
        pc,
        grasp=None,
        concat_option="channel_wise",
        rotz=True,
        feature_2=False,
        train=True,
    ):
        # separate policy and value networks
        input_features = pc
        input_features_vis = input_features
        if input_features.shape[-1] != 1024: # hand points included
            input_features = input_features[..., 6:]

        input_features = (
            input_features[:, : self.critic_input_dim].contiguous()
            if feature_2
            else input_features[:, : self.policy_input_dim].contiguous()
        )
        object_grasp_pc = input_features.transpose(1, -1)[..., :3].contiguous()

        if feature_2:
            return (
                self.encode(self.value_encoder, object_grasp_pc, input_features),
                input_features_vis,
            )
        z = self.encode(self.encoder, object_grasp_pc, input_features)
        return z, input_features_vis

# https://github.com/pranz24/pytorch-soft-actor-critic/
class QNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        pixel_supervision=False,
        extra_pred_dim=0,
    ):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.extra_pred_dim = extra_pred_dim

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        if self.extra_pred_dim > 0:
            self.linear7 = nn.Linear(num_inputs, hidden_dim)
            self.linear8 = nn.Linear(hidden_dim, hidden_dim)
            self.extra_pred = nn.Linear(hidden_dim, self.extra_pred_dim)
        self.apply(weights_init_)

    def forward(self, state, action=None):
        x3 = None
        if action is not None:
            xu = torch.cat([state, action], 1)
        else:
            xu = state
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        if self.extra_pred_dim:
            x3 = F.relu(self.linear7(state))
            x3 = F.relu(self.linear8(x3))
            x3 = self.extra_pred(x3)
            if self.extra_pred_dim == 7: # normalize quaternion
                x3 = torch.cat((F.normalize(x3[:, :4], p=2, dim=-1), x3[:, 4:]), dim=-1)
        return x1, x2, x3


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        extra_pred_dim=0,
        uncertainty=False,
    ):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty = uncertainty

        self.extra_pred_dim = extra_pred_dim
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.extra_pred = nn.Linear(hidden_dim, self.extra_pred_dim)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        self.action_space = action_space

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0).cuda()
            self.action_bias = torch.tensor(0.0).cuda()
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            ).cuda()
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            ).cuda()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        extra_pred = self.extra_pred(x)
        if self.extra_pred_dim == 7:
            extra_pred = torch.cat(
                (F.normalize(extra_pred[:, :4], p=2, dim=-1), extra_pred[:, 4:]), dim=-1
            )
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, extra_pred

    def sample(self, state):
        mean, log_std, extra_pred = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        if self.action_space is not None:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            y_t = x_t
            action = x_t

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        if self.action_space is not None:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean, log_prob, action, extra_pred


    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
