# 打开存放图片的文件夹，然后遍历文件名，把文件名字， label 还有 文件夹名写入data.txt文件中。

import os


def make_txt(root, file_name, label):
    path = os.path.join(root, file_name)

    data = os.listdir(path)

    f = open(root + '\\' + 'data.txt', 'a')

    for line in data:
        f.write(line + ' ' + str(label) + ' ' + file_name + '\n')
    f.close()


path = r'D:\code\platform\Deep-Learning-Framework\Data'

# 调用函数生成两个文件夹下的txt文件
make_txt(path, file_name='cat', label=0)
