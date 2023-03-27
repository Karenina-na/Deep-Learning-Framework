import matplotlib.pylab as plt
import numpy as np


# 画 train-test 准确率与损失曲线
def plot_model_img(train_y, test_y, loss):
    x = np.arange(len(train_y))
    """ Plot image """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("accuracy")
    plt.plot(x, train_y, 'r-', label='train')
    plt.legend(loc='upper left')
    plt.plot(x, test_y, 'b-', label='test')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.title("loss")
    plt.plot(x, loss, 'r-', label='loss')
    plt.legend(loc='upper left')
    plt.show()


# 画生成对抗网络的损失曲线
def plot_gan_img(d_loss, g_loss, glo_loss, path="./"):
    x = np.arange(len(d_loss))
    """ Plot image """
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Discriminator and Generator loss")
    plt.plot(x, d_loss, 'r-', label='d_loss')
    plt.plot(x, g_loss, 'b-', label='g_loss')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.title("glo_loss")
    plt.plot(x, glo_loss, 'g-', label='glo_loss')
    plt.legend(loc='upper left')
    plt.show()
