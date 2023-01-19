import torch
import torch.multiprocessing as mp
import numpy as np
import gym
import os
from Agent.A3C_Agent import Net

os.environ["OMP_NUM_THREADS"] = "1"

# A3C
UPDATE_GLOBAL_ITER = 5  # update global network every 5 episodes
GAMMA = 0.9  # reward discount
MAX_EP = 1800  # maximum episode
PARAMETER_NUM = mp.cpu_count()  # the number of parameters

env = gym.make('CartPole-v1', render_mode='rgb_array')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
env.close()


class SharedAdam(torch.optim.Adam):
    """
    The following two functions are copied from the original pytorch source code.
    """

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


def v_wrap(np_array, dtype=np.float32):
    """
    Wrap the numpy array into a torch tensor
    :param np_array:
    :param dtype:
    :return:
    """
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    """
    Record the result
    :param global_ep:   the global episode
    :param global_ep_r: the global episode reward
    :param ep_r:        the episode reward
    :param res_queue:   the result queue
    :param name:        the name of the process
    :return:            None
    """
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


def push_and_pull_net(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    """
    Push the gradients to the global network and pull the parameters from the global network
    :param opt: the optimizer
    :param lnet:    the local network
    :param gnet:    the global network
    :param done:    whether the episode is done
    :param s_:  the next state
    :param bs:  the state
    :param ba:  the action
    :param br:  the reward
    :param gamma:   the discount factor
    :return:    None
    """
    if done:
        v_s_ = 0.  # terminal
    else:
        # get the value of the next state, add the batch dimension
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    # calculate the advantage   R(t) + gamma * V(s_(t+1)) - V(s_t)
    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # calculate the loss
    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients
    opt.zero_grad()
    loss.backward()
    # push the gradients to the global network
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        """
        :param gnet:            global network
        :param opt:             optimizer
        :param global_ep:       global episode
        :param global_ep_r:     global episode reward
        :param res_queue:       result queue
        :param name:            worker name
        """
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    def run(self):
        """
        run
        :return:    None
        """
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s, _ = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':  # only one worker can show the environment
                    self.env.render()
                # choose action
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                # take action
                s_, r, done, _, _ = self.env.step(a)
                # record the transition
                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull_net(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

        print("Worker {} finished".format(self.name))


if __name__ == "__main__":
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer

    # global_ep, global_ep_r, res_queue are used to record the result,
    # global_ep is the global episode, global_ep_r is the global episode reward,
    # res_queue is used to record the result
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(PARAMETER_NUM)]

    # start training
    [w.start() for w in workers]

    # record episode reward to plot
    res = []

    # change structure of the result queue
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    # wait for all workers to finish
    [w.join() for w in workers]

    print("Finished")

    # show training result
    env = gym.make('CartPole-v1', render_mode='human').unwrapped
    s, _ = env.reset()
    while True:
        env.render()
        a = gnet.choose_action(v_wrap(s[None, :]))
        s, r, done, _, _ = env.step(a)
        if done:
            break

    # plot the result
    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
