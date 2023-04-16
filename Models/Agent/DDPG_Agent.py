import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
import os


# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state, done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done))

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)


# ------------------------------------- #
# 策略网络
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, action_bound):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        self.action_bound = action_bound
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hidden]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hidden]-->[b,n_actions]
        x = torch.tanh(x)  # 将数值调整到 [-1,1]
        x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x


# ------------------------------------- #
# 价值网络
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(QValueNet, self).__init__()
        #
        self.fc1 = nn.Linear(n_states + n_actions, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hidden]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hidden]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x


# ------------------------------------- #
# 算法主体
# ------------------------------------- #

class Agent(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device, path=None, type="train"):
        super(Agent, self).__init__()
        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_hidden, n_actions, action_bound).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_hidden, n_actions).to(device)
        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_hidden, n_actions, action_bound).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_hidden, n_actions).to(device
                                                                         )
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.device = device
        self.type = type

        # 模型加载与保存
        self.model_path = path
        if self.model_path is not None:
            if os.path.exists(self.model_path + "/DDPG.pth"):
                self.load_state_dict(torch.load(self.model_path + "/DDPG.pth"))
                print("load model from {}".format(self.model_path + "/DDPG.pth"))
            else:
                print("model not exists")
        else:
            print("no model to load")

    # 动作选择
    def take_action(self, state):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(state).detach()
        # 给动作添加噪声，增加搜索
        if self.type == "train":
            action = action + self.sigma * np.random.randn(self.n_actions)
        return action.numpy()

    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    # 训练
    def update(self, transition_dict):
        # 从训练集中取出数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        action = torch.tensor(transition_dict['action'], dtype=torch.float).to(self.device)  # [b,n_actions]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,next_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        # 策略目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_q_actions = self.target_actor(next_states)
        # 价值目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_value = self.target_critic(next_states, next_q_actions)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_value * (1 - dones)

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, action)

        # 预测值和目标值之间的均方差损失 loss=(q_targets-q_values)^2
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(states, actor_q_values)
        # 计算损失 loss=-score
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)

    def save_model(self):
        # 删除模型
        if os.path.exists(self.model_path + "/DDPG.pth"):
            os.remove(self.model_path + "/DDPG.pth")
        # 保存模型
        torch.save(self.state_dict(), self.model_path + "/DDPG.pth")


if __name__ == "__main__":
    # 存储大小
    buffer = ReplayBuffer(10000)
    s = np.random.randn(3)
    a = np.random.randn(4)
    r = np.random.randn(1)
    s_ = np.random.randn(3)
    d = np.random.randn(1)
    buffer.add(s, a, r, s_, d)
    print(buffer.sample(1))

    # 动作维度不等于动作个数
    # 环境状态，隐藏层，动作维度，动作范围。
    policy = PolicyNet(3, 128, 4, 1)
    s = torch.randn(2, 3)
    print(policy(s))

    # 环境状态，隐藏层，动作维度
    value = QValueNet(3, 128, 4)
    s = torch.randn(2, 3)
    a = torch.randn(2, 4)
    print(value(s, a))

    # 环境状态，隐藏层，动作维度，动作范围，高斯噪声的标准差，策略学习率，价值学习率，目标网络的软更新参数，折扣因子，device
    agent = Agent(3, 128, 4, 1, 0.2, 0.001, 0.001, 0.01, 0.99, 'cpu')
    s = np.random.randn(3)
    print(agent.take_action(s))

    transition_dict = {
        'states': np.random.randn(32, 3),
        'action': np.random.randn(32, 4),
        'rewards': np.random.randn(32, 1),
        'next_states': np.random.randn(32, 3),
        'dones': np.random.randn(32, 1)
    }
    agent.update(transition_dict)
