import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.optim as optim
from Models.DCGAN.Discriminator import Discriminator
from Models.DCGAN.Generator import Generator
from util.plot.plot_loss_accuracy import plot_gan_img
from torch.utils.data import DataLoader
from DataProcess.CatImageLoad import MyDataset, transforms

# 超参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

z_dim = 100
batch_size = 512
generator_learning_rate = 0.0002
discriminator_learning_rate = 0.0002
total_epochs = 2000
discriminator_Loss_Limit = 0.4
generator_more_epoch = 2

# 数据
# 加载需要生成的数据集
# 加载数据
path = r'../../Data'
dataset = MyDataset(path, transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# 使用均值为0，方差为0.02的正态分布初始化权重
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim).to(device)

# # 多卡 GPU
# discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1])
# generator = torch.nn.DataParallel(generator, device_ids=[0, 1])

# 使用均值为0，方差为0.02的正态分布初始化神经网络
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)
ones = torch.ones(batch_size, 1).to(device)
zeros = torch.zeros(batch_size, 1).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=[0.5, 0.999])

# 随机产生100个向量，用于生成效果图
fixed_z = torch.randn([100, z_dim]).to(device)

# 损失
discriminator_loss = []
generator_loss = []
global_loss = []

# 开始训练，一共训练total_epochs
for epoch in range(total_epochs):

    # 画图数据
    discriminator_epoch_loss = 0
    generator_epoch_loss = 0
    global_epoch_loss = 0

    count = len(dataloader)

    # 在训练阶段，把生成器设置为训练模型；对应于后面的，在测试阶段，把生成器设置为测试模型
    generator = generator.train()

    # 训练一个epoch
    for i, data in enumerate(dataloader):
        # 加载真实数据，不加载标签
        real_images, _ = data
        real_images = real_images.to(device)

        # 用正态分布中采样batch_size个噪声，然后生成对应的图片
        z = torch.randn([batch_size, z_dim]).to(device)
        fake_images = generator(z)

        # 计算判别器损失，并优化判别器
        fake_loss = bce(discriminator(fake_images.detach()), zeros)
        real_loss = bce(discriminator(real_images), ones)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()

        # 如果判别器Loss小于0.4，就不更新判别器
        if d_loss > discriminator_Loss_Limit:
            d_loss.backward()
            d_optimizer.step()

        g_epoch_loss = 0
        # 多次训练生成器,每训练一次判别器，训练generator_epoch次生成器
        for j in range(generator_more_epoch):
            # 用正态分布中采样batch_size个噪声，然后生成对应的图片
            z = torch.randn([batch_size, z_dim]).to(device)
            fake_images = generator(z)

            # 计算生成器损失，并优化生成器
            g_loss = bce(discriminator(fake_images), ones)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            with torch.no_grad():
                g_epoch_loss += g_loss.item()

        # 保存损失
        with torch.no_grad():
            discriminator_epoch_loss += d_loss.item()
            generator_epoch_loss += g_epoch_loss / generator_more_epoch
            global_epoch_loss += d_loss.item() + g_epoch_loss / generator_more_epoch

    # 把生成器设置为测试模型，生成效果图并保存
    generator = generator.eval()
    with torch.no_grad():

        dLoss = discriminator_epoch_loss / count
        gLoss = generator_epoch_loss / count
        gloLoss = global_epoch_loss / count

        # 生成效果图
        discriminator_loss.append(dLoss)
        generator_loss.append(gLoss)
        global_loss.append(gloLoss)

        print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, glo_loss {:.4f}'.format(total_epochs, epoch, dLoss, gLoss,
                                                                                      gloLoss))

# 画图
plot_gan_img(discriminator_loss, generator_loss, global_loss, "../../Result")

# 保存生成器与判别器模型
torch.save(generator.state_dict(), './Result/Model/generator.pth')
torch.save(discriminator.state_dict(), './Result/Model/discriminator.pth')
