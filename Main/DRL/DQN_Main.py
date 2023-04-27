import gym
import numpy as np
import torch
import torch.nn as nn
import random
from Models.Agent.DQN_Agent import Agent


def train():
    env = gym.make(GAME, render_mode='rgb_array')

    s, info = env.reset()

    n_state = len(s)
    n_action = env.action_space.n

    """Generate agents"""

    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train', model_path=model_path)

    """Main Training Loop"""

    REWARD_BUFFER = np.zeros(shape=n_episode)

    for episode_i in range(n_episode):

        # 一次游戏的总reward
        episode_reward = 0

        for step_i in range(n_time_step):

            # 选择随机动作的概率
            epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY],
                                [EPSILON_START, EPSILON_END])  # interpolation

            # 概率epsilon随机选择动作，概率1-epsilon选择最优动作 online网络
            random_sample = random.random()
            if random_sample <= epsilon:
                a = env.action_space.sample()
            else:
                a = agent.online_net.act(s)

            # 执行动作
            s_, r, done, trunk, info = env.step(a)  # trunk,info will not be used

            # 记录经验
            agent.memo.add_memo(s, a, r, done, s_)

            # 更新状态
            s = s_

            # 记录奖励
            episode_reward += r

            if done:
                s, info = env.reset()

                # 记录奖励
                REWARD_BUFFER[episode_i] = episode_reward
                break

            # 抽取经验
            batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q

            # 计算 target Q(t+1) target网络
            target_q_values = agent.target_net(batch_s_)

            # 计算 target max Q(t+1)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            # 计算 target Q(t)
            targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

            # 计算 online Q(t) online网络
            q_values = agent.online_net(batch_s)

            # 按照 batch_a 选择 Q(t) 中的动作 维度为行
            a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

            # 计算 loss
            loss = nn.functional.smooth_l1_loss(a_q_values, targets)

            # 更新 online 网络
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        # 更新 target 网络
        if episode_i % TARGET_UPDATE_FREQUENCY == 0 and episode_i != 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())

            print("Episode: {}".format(episode_i))
            print("Avg Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))

            # 如果奖励到达一定值，演示
            if episode_i != 0 and np.mean(REWARD_BUFFER[:episode_i]) >= DEMO_REWARD:
                count = 0
                env_e = gym.make(GAME, render_mode='human')
                s, info = env_e.reset()
                # 演示
                step = 0
                while True:
                    a = agent.online_net.act(s)
                    s, r, done, trunk, info = env_e.step(a)
                    count += r
                    step += 1
                    env_e.render()
                    if done or step >= n_time_step:
                        env_e.reset()
                        print("#" * 50)
                        print("Finished Reward: ", count)
                        print("Step Number: ", step)
                        print("-" * 50)
                        env_e.close()
                        break

    agent.save_model()


def test():
    """Generate agents"""
    env = gym.make(GAME, render_mode='human')
    s, info = env.reset()
    n_state = len(s)
    n_action = env.action_space.n

    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='test', model_path=model_path)

    reward_all = 0

    # 演示
    step = 0
    while True:
        a = agent.online_net.act(s)
        s, r, done, trunk, info = env.step(a)
        reward_all += r
        step += 1
        env.render()
        if done or step >= n_time_step:
            env.reset()
            print("#" * 50)
            print("Finished Reward: ", reward_all)
            print("Step Number: ", step)
            print("-" * 50)
            break

    env.close()


# 探索率初始
EPSILON_START = 1.0
# 探索率结束
EPSILON_END = 0.02
# 探索率衰减率
EPSILON_DECAY = 100000
# Target Network 更新频率
TARGET_UPDATE_FREQUENCY = 10
# 平均reward到达多少演示
DEMO_REWARD = 100
# 训练次数
n_episode = 200
# 每次训练的最大步数
n_time_step = 400
# 游戏
GAME = "LunarLander-v2"
# model
model_path = "../../Result/checkpoints"

if __name__ == '__main__':
    # train()
    test()
