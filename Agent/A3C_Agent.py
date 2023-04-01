import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from multiprocessing import Process, Queue


# Actor模块
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # 第一层全连接层，输入为状态
        self.fc2 = nn.Linear(64, 128)  # 第二层全连接层
        self.fc3 = nn.Linear(128, action_dim)  # 第三层全连接层，输出为动作空间的维度

    def forward(self, state):
        x = F.relu(self.fc1(state))  # 通过ReLU激活函数处理第一层输出
        x = F.relu(self.fc2(x))  # 通过ReLU激活函数处理第二层输出
        action_probs = F.softmax(self.fc3(x), dim=-1)  # 通过Softmax激活函数处理第三层输出，得到动作的概率分布
        return action_probs


# Critic模块
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # 第一层全连接层，输入为状态
        self.fc2 = nn.Linear(64, 128)  # 第二层全连接层
        self.fc3 = nn.Linear(128, 1)  # 第三层全连接层，输出为1维度的值，即V(s)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # 通过ReLU激活函数处理第一层输出
        x = F.relu(self.fc2(x))  # 通过ReLU激活函数处理第二层输出
        state_value = self.fc3(x)  # 最后一层输出为状态值函数V(s)
        return state_value


# Agent
class Agent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor_local = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_local = Critic(state_dim)
        self.critic_target = Critic(state_dim)
        self.gamma = gamma

        # 确保目标网络与本地网络具有相同的权重
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # 将目标网络设置为评估模式
        self.actor_target.eval()
        self.critic_target.eval()

        # 为演员和评论家网络定义优化器
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_local(state)
        action_probs = action_probs.cpu().data.numpy()
        action = np.random.choice(np.arange(len(action_probs.squeeze())), p=action_probs.squeeze())
        return action

    def learn(self, states, actions, rewards, next_states):
        # 将输入转换为张量，判断数据类型
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states)
        states = states.float()
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions)
        actions = actions.long()
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.from_numpy(rewards)
        rewards = rewards.float()
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.from_numpy(next_states)
        next_states = next_states.float()

        # 计算TD误差
        td_targets = rewards + self.gamma * self.critic_target(next_states)
        td_errors = td_targets - self.critic_local(states)

        # 更新评论家网络
        critic_loss = (td_errors ** 2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer_critic.step()

        # 更新演员网络
        advantages = td_targets - self.critic_local(states).detach()
        actor_loss = -(self.critic_local(states).detach() * advantages *
                       self.actor_local(states).gather(1, actions)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # 更新目标网络
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

    @staticmethod
    def soft_update(local_model, target_model, tau=0.01):
        # tau是软更新的参数, tau越小, target_model更新越慢
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    agent = Agent(10, 5)
    s = torch.randn(64, 10)
    a = torch.randint(5, [64, 1])  # 采取的动作
    r = torch.randn(64, 1)
    s_ = torch.randn(64, 10)
    print("actor_local: ", agent.actor_local(s).shape)
    print("actor_target: ", agent.actor_target(s).shape)
    print("critic_local: ", agent.critic_local(s).shape)
    print("critic_target: ", agent.critic_target(s).shape)
    agent.learn(s, a, r, s_)
    print("actor_local one gradient: ", agent.actor_local(s).shape)
    print("actor_target one gradient: ", agent.actor_target(s).shape)
    print("critic_local: one gradient", agent.critic_local(s).shape)
    print("critic_target: one gradient", agent.critic_target(s).shape)
