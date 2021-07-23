import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(ActorCritic, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.data = []

        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, out_dim)
        self.critic = nn.Linear(128, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def pi(self, x, softmax_dim=0):
        x = self.backbone(x)
        x = self.actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.backbone(x)
        v = self.critic(x)
        return v
