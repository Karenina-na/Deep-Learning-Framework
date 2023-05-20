## 常见pytorch卷积层
```blank
nn.Linear：全连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)

nn.Flatten：压平层，用于将多维张量样本压成一维张量样本。

nn.BatchNorm1d：一维批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。可以用afine参数设置该层是否含有可以训练的参数。

nn.BatchNorm2d：二维批标准化层。

nn.BatchNorm3d：三维批标准化层。

nn.Dropout：一维随机丢弃层。一种正则化手段。

nn.Dropout2d：二维随机丢弃层。

nn.Dropout3d：三维随机丢弃层。

nn.Threshold：限幅层。当输入大于或小于阈值范围时，截断之。

nn.ConstantPad2d： 二维常数填充层。对二维张量样本填充常数扩展长度。

nn.ReplicationPad1d： 一维复制填充层。对一维张量样本通过复制边缘值填充扩展长度。

nn.ZeroPad2d：二维零值填充层。对二维张量样本在边缘填充0值.

nn.GroupNorm：组归一化。一种替代批归一化的方法，将通道分成若干组进行归一。不受batch大小限制，据称性能和效果都优于BatchNorm。

nn.LayerNorm：层归一化。较少使用。

nn.InstanceNorm2d: 样本归一化。较少使用
```