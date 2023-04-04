import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from multiprocessing import Process, Queue


def set_init(layers):
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
    def __init__(self, s_dim, a_dim, GAMMA):
        super(Agent, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.GAMMA = GAMMA
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
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

    def choose_action(self, s):
        """
        根据状态选择动作
        :param s:   状态 [state_dim]
        :return:    动作 [action_dim]
        """
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state, actions, rewards):
        """
        计算损失函数
        :param state:   状态 [batch_size, state_dim]
        :param actions: 动作分布 [batch_size, action_dim]
        :param rewards: 多步奖励 [batch_size, [reward1, reward2, ...]
        :return: actor_loss, critic_loss
        """
        self.train()

        # 计算当前状态的价值
        logits, values = self.forward(state)

        # 计算累计奖励
        returns = []
        for i in range(len(rewards)):
            Gt = 0  # 未来奖励
            pw = 0  # 未来奖励衰减权重
            for r in rewards[i:]:
                Gt = Gt + self.GAMMA ** pw * r
                pw = pw + 1
            returns.append(Gt)
        returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)

        # 计算 advantage
        advantages = []
        for i in range(len(rewards)):
            advantages.append(returns[i] - values[i])

        # 计算损失函数
        actor_loss = []
        critic_loss = []
        for logit, advantage, action in zip(logits, advantages, actions):
            m = self.distribution(F.softmax(logit, dim=0))
            log_prob = m.log_prob(action)
            critic_loss.append(F.smooth_l1_loss(values, returns))
            actor_loss.append(-log_prob * advantage)

        # 总损失函数
        total_loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

        return torch.stack(actor_loss).sum().detach().numpy(), torch.stack(
            critic_loss).sum().detach().numpy(), total_loss


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

