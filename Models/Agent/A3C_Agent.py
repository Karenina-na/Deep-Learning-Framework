import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from multiprocessing import Process, Queue


def set_init(layers: list):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Agent(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, GAMMA: float = 0.9, model_path=None):
        super(Agent, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.GAMMA = GAMMA
        # policy network
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        # value network
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        # init
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

        self.model_path = model_path
        if self.model_path is not None:
            if os.path.exists(model_path + "/A3C.pth"):
                self.load_state_dict(torch.load(model_path + "/A3C.pth"))
                print("load model from {}".format(model_path + "/A3C.pth"))
            else:
                print("model not exists")
        else:
            print("no model to load")

    def forward(self, x: torch.Tensor):
        """
        前向传播
        :param x:   状态 [batch_size, state_dim]
        :return:    动作分布 [batch_size, action_dim], 价值函数 [batch_size, 1]
        """
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, state: torch.Tensor):
        """
        根据状态选择动作
        :param state:   状态 [state_dim]
        :return:    动作 [action_dim]
        """
        self.eval()
        logits, _ = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state: torch.Tensor, actions: torch.Tensor, target: torch.Tensor):
        """
        计算损失函数
        :param state:   状态 [batch_size, state_dim]
        :param actions: 动作分布 [batch_size, action_dim]
        :param target: target [batch_size, 1]
        :return: actor_loss, critic_loss
        """
        self.train()
        # 计算当前状态的价值
        logits, values = self.forward(state)

        # 计算累计奖励    R = r + GAMMA * r + GAMMA^2 * r + ...
        returns = []
        for i in range(len(target)):
            Gt = 0  # 未来奖励
            pw = 0  # 未来奖励衰减权重
            for r in target[i:]:
                Gt = Gt + self.GAMMA ** pw * r
                pw = pw + 1
            returns.append(Gt)
        returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)

        # 计算 advantage
        advantages = []
        for i in range(len(target)):
            advantages.append(returns[i] - values[i])

        # 计算损失函数
        actor_loss = []
        critic_loss = []
        for logit, advantage, action in zip(logits, advantages, actions):
            m = self.distribution(F.softmax(logit, dim=0))
            log_prob = m.log_prob(action)
            critic_loss.append(F.smooth_l1_loss(values, returns))
            actor_loss.append(-log_prob * advantage)

        return torch.stack(actor_loss).sum(), torch.stack(critic_loss).sum()

    def save_model(self):
        if self.model_path is not None:
            torch.save(self.state_dict(), self.model_path + "/A3C.pth")
            print("model saved to {}".format(self.model_path + "/A3C.pth"))
        else:
            print("no model to save")


if __name__ == "__main__":
    batch_size = 5
    s = torch.rand([batch_size, 5], dtype=torch.float32)
    a = torch.rand([batch_size, 10], dtype=torch.float32)
    r = torch.rand([batch_size, 1], dtype=torch.float32)
    print("state shape:", s.shape)
    print("action prob shape:", a.shape)
    print("reward shape:", r.shape)
    agent = Agent(s_dim=5, a_dim=10, GAMMA=0.9)
    print(agent(s))
