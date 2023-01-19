import torch
import torch.nn as nn
import torch.nn.functional as F


def set_init(layer):
    """
    initialize weights
    :param layer:   layer
    :return:    None
    """
    nn.init.normal_(layer.weight, mean=0., std=0.1)
    nn.init.constant_(layer.bias, 0.)


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim  # state dimension
        self.a_dim = a_dim  # action dimension

        # policy
        self.policy1 = nn.Linear(s_dim, 128)
        self.policy2 = nn.Tanh()
        self.policy3 = nn.Linear(128, a_dim)
        # value
        self.value1 = nn.Linear(s_dim, 128)
        self.value2 = nn.Tanh()
        self.value3 = nn.Linear(128, 1)

        # initialize weights
        set_init(self.policy1)
        set_init(self.policy3)
        set_init(self.value1)
        set_init(self.value3)

        self.distribution = torch.distributions.Categorical  # action distribution

    def forward(self, x):
        # policy
        actions = self.policy3(self.policy2(self.policy1(x)))

        # value
        values = self.value3(self.value2(self.value1(x)))

        return actions, values

    def choose_action(self, state):
        """
        choose action
        :param state:   state
        :return:    action
        """
        self.eval()
        actions, _ = self.forward(state)
        # the probability of the action
        prob = F.softmax(actions, dim=1).data
        # calculate distribution of the action
        m = self.distribution(prob)
        return m.sample().numpy().reshape(1)[0]

    def loss_func(self, state, action, v_s):
        """
        loss function
        :param state:   state
        :param action:   action
        :param v_s:     value
        :return:    loss
        """
        self.train()
        # logits: the number of each action，values: the value of the state
        actions, values = self.forward(state)

        # calculate TD error
        td = v_s - values
        # critic loss
        c_loss = td.pow(2)

        # the probability of the action
        probs = F.softmax(actions, dim=1)
        # calculate distribution of the action
        m = self.distribution(probs)
        # PDF of the action, log_prob: -log(PDF),
        a_loss = -m.log_prob(action) * td.detach().squeeze()

        # total loss, mean of the loss
        total_loss = (c_loss + a_loss).mean()
        return total_loss


if __name__ == '__main__':
    s = torch.randn(1, 4)
    a = torch.randint(0, 3, [1, 1])
    n = Net(4, 3)
    logits, values = n(s)
    print(logits.shape)
    print(values.shape)
    print(n.choose_action(s))
    print(n.loss_func(s, a, values))