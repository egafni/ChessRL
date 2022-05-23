from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from chessrl.env import ChessEnv
from chessrl.model import ActorCritic


@dataclass
class TrainerParams:
    learning_rate = .001
    gamma: float = 0.98
    lmbda: float = 0.95
    eps_clip: float = 0.1
    sgd_iters: int = 3
    T_horizon: int = 100
    print_interval: int = 20


class Trainer:
    def __init__(self, params: TrainerParams, agent: ActorCritic):
        self.params = params
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=params.learning_rate)
        self.data = []

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
            td_target = r + self.params.gamma * self.agent.v(s_prime) * done_mask
            delta = td_target - self.agent.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.params.gamma * self.params.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            s = s.to(self.agent.device)
            pi = self.agent.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.params.eps_clip, 1 + self.params.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.agent.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def train(self, env: ChessEnv):
        score = 0.0
        for episode_number in range(10000):
            s = env.reset()
            s = s[None, :, :, :]  # add batch dimension

            done = False
            while not done:
                for t in range(self.params.T_horizon):
                    pi = self.agent.pi(torch.from_numpy(s))[0]  # batch size is 1 select first element

                    pi = env.mask_illegal_moves_from_pi_in_place(pi)
                    m = Categorical(pi)
                    a = m.sample().item()

                    s_prime, r, done, info = env.step(a)

                    self.put_data((s, a, r / 100.0, s_prime, pi[a].item(), done))
                    s = s_prime

                    score += r
                    if done:
                        break

                self.train_net()

            if episode_number % 1 == 0:
                env.opponent.load_state_dict(deepcopy(self.agent.state_dict))

            if episode_number % self.params.print_interval == 0 and episode_number != 0:
                print("# of episode :{}, avg score : {:.1f}".format(episode_number, score / self.params.print_interval))
                score = 0.0
