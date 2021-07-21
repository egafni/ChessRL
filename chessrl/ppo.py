from dataclasses import dataclass

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from chessrl.env import ChessEnv


@dataclass
class PPOParams:
    in_dim: int
    out_dim: int
    learning_rate = .001
    gamma: float = 0.98
    lmbda: float = 0.95
    eps_clip: float = 0.1
    sgd_iters: int = 3
    T_horizon: int = 100
    print_interval: int = 20


class PPOAgent(nn.Module):
    def __init__(self, params: PPOParams):
        super(PPOAgent, self).__init__()
        self.params = params
        self.data = []

        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(params.in_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, self.params.out_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=params.learning_rate)

    def pi(self, x, softmax_dim=0):
        x = self.backbone(x)
        x = self.actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.backbone(x)
        v = self.critic(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.params.sgd_iters):
            td_target = r + self.params.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.params.gamma * self.params.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.params.eps_clip, 1 + self.params.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def learn(self, env: ChessEnv, on_episode_learned: callable):
        score = 0.0
        for episode_number in range(10000):
            s = env.reset()
            done = False
            while not done:
                for t in range(self.params.T_horizon):
                    prob = self.pi(torch.from_numpy(s).float().cuda())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)

                    self.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                    s = s_prime

                    score += r
                    if done:
                        break

                self.train_net()

            on_episode_learned(episode_number)

            if episode_number % self.params.print_interval == 0 and episode_number != 0:
                print("# of episode :{}, avg score : {:.1f}".format(episode_number, score / self.params.print_interval))
                score = 0.0

