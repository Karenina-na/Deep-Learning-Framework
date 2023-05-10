def mobilevit_xxs(img_size=(256, 256), num_classes=1000):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes, expansion=2)


def mobilevit_xs(img_size=(256, 256), num_classes=1000):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes)


def mobilevit_s(img_size=(256, 256), num_classes=1000):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes)