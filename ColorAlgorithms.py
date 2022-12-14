import torch
from torchvision.transforms.functional import adjust_saturation, adjust_hue


def Gray_World(img):
    m = img.mean(-2, True).mean(-1, True)
    img = img / torch.clamp(m, min=1e-3)
    ma = img.max(-1, True).values.max(-2, True).values.max(-3, True).values
    return img / torch.clamp(ma, min=1e-3)


def MaxRGB(img):
    maxs = img.max(-1, True).values.max(-2, True).values.max(-3, True).values
    return img / torch.clamp(maxs, min=1e-3)


def saturation(img, param):
    return adjust_saturation(img, param)


def hue(img, param):
    return adjust_hue(img, param)
