# Paper with Code

## 1. Introduction

This Markdown file is used to record the code of the paper I read.

## 2. Paper with Code

###  Computer Vision Backbone
| Model                        |                                                                          Paper                                                                          |                      Code                      |
|------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------:|
| LeNet-5                      |                         [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)                          |            [code](./Models/LetNet)             |
| ResNet                       |                                    [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)                                     |            [code](./Models/ResNet)             |
| ResNeSt                      |                                          [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)                                          |            [code](./Models/ResNeSt)            |
| GoogLeNet                    |                                            [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)                                            |   [code](./Models/Inception/GoogLeNet_2d.py)   |
| Inception-v4                 |                  [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)                  | [code](./Models/Inception/GoogLe_ResNet_1d.py) |
| MobileNet v1                 |                 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)                  |   [code](./Models/MobileNet/MobileV1_2d.py)    |
| MobileNet v2                 |                               [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)                                |   [code](./Models/MobileNet/MobileV2_2d.py)    |
| MobileNet v3                 |                                              [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)                                              |   [code](./Models/MobileNet/MobileV3_2d.py)    |
| ConvNeXt                     |                                      [ConvNeXt: A Convolutional Neural Network](https://arxiv.org/abs/2201.03545)                                       |           [code](./Models/ConvNeXt)            |
| Vision Transformer           |                     [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)                      |    [code](./Models/VisionTransformer_DeiT)     |
| Distilled Vision Transformer |                     [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)                     |    [code](./Models/VisionTransformer_DeiT)     |
| Swin Transformer v1          |                       [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)                       |    [code](./Models/SwinTransformer/Swin_v1)    |
| Swin Transformer v2          |                               [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)                               |    [code](./Models/SwinTransformer/Swin_v2)    |
| Mobile Vision Transformer v1 |                  [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)                   |    [code](./Models/MobileViT/MobileViT_v1)     |
| Mobile Vision Transformer v2 |                               [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)                               |    [code](./Models/MobileViT/MobileViT_v2)     |
| Mobile Vision Transformer v3 | [MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features](http://arxiv.org/abs/2209.15159) |    [code](./Models/MobileViT/MobileViT_v3)     |

### Image Generation
| Model |                                                              Paper                                                               |          Code          |
|-------|:--------------------------------------------------------------------------------------------------------------------------------:|:----------------------:|
| GAN   |                                [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                                |  [code](./Models/GAN)  |
| DCGAN | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) | [code](./Models/DCGAN) |

### Object Detection
| Model  |                                                             Paper                                                              |              Code               |
|--------|:------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|
| YOLO 5 |                            [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)                            | [code](./Models/Yolo/Yolo%20v5) |
| YOLO 7 | [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) | [code](./Models/Yolo/Yolo%20v7) |
| YOLO 8 |                                           [[website]](https://docs.ultralytics.com/)                                           | [code](./Models/Yolo/Yolo%20v8) |

### Natural Language Processing
| Model       |                                      Paper                                      |             Code             |
|-------------|:-------------------------------------------------------------------------------:|:----------------------------:|
| LSTM        | [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) |    [code](./Models/LSTM)     |
| Transformer |          [Attention Is All You Need](https://arxiv.org/abs/1706.03762)          | [code](./Models/Transformer) |


### Deep Reinforcement Learning
| Model      |                                          Paper                                           |                Agent                 |              Main               |
|------------|:----------------------------------------------------------------------------------------:|:------------------------------------:|:-------------------------------:|
| DQN        |    [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)     | [code](./Models/Agent/DQN_Agent.py)  | [code](./Main/DRL/DQN_Main.py)  |
| Double DQN |  [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  | [code](./Models/Agent/DQN_Agent.py)  | [code](./Main/DRL/DQN_Main.py)  |
| DDPG       | [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  | [code](./Models/Agent/DDPG_Agent.py) | [code](./Main/DRL/DDPG_Main.py) |
| PPO        |       [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)        | [code](./Models/Agent/PPO_Agent.py)  | [code](./Main/DRL/PPO_Main.py)  |
| A3C        | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) | [code](./Models/Agent/A3C_Agent.py)  | [code](./Main/DRL/A3C_Main.py)  |