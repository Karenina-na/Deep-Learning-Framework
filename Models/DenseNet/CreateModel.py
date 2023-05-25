from Models.DenseNet.DenseNet import DenseNet


def DenseNet121(num_classes):
    return DenseNet(blocks=(6, 12, 24, 16), num_classes=num_classes)


def DenseNet169(num_classes):
    return DenseNet(blocks=(6, 12, 32, 32), num_classes=num_classes)


def DenseNet201(num_classes):
    return DenseNet(blocks=(6, 12, 48, 32), num_classes=num_classes)


def DenseNet264(num_classes):
    return DenseNet(blocks=(6, 12, 64, 48), num_classes=num_classes)
