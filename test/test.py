import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
for i in range(10000):
    x.append(i ** 2)
    y.append(2 *i ** 2)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12
plt.figure(figsize=(10, 8), dpi=100)
plt.plot(x, linestyle='--', label="本系统")
plt.plot(y, linestyle='-', label="传统系统")
plt.xlabel("存储对象个数", fontsize=18)
plt.ylabel("存储对象时延", fontsize=18)
plt.title("存储对象耗费时间对比", fontsize=20)
plt.legend()
plt.show()
