import torch
import torch.utils
import torch.utils.data
from ColorAlgorithms import Gray_World, MaxRGB, saturation, hue
from torchvision import transforms
from PIL import ImageFilter


def select(img, act):
    if act == 0:
        return gamma_corr(img, 0.6, 0)
    elif act == 1:
        return gamma_corr(img, 0.6, 1)
    elif act == 2:
        return gamma_corr(img, 0.6, 2)
    elif act == 3:
        return gamma_corr(img, 1.1, 0)
    elif act == 4:
        return gamma_corr(img, 1.1, 1)
    elif act == 5:
        return gamma_corr(img, 1.1, 2)
    elif act == 6:
        return gamma_corr(img, 0.6)
    elif act == 7:
        return gamma_corr(img, 1.1)
    elif act == 8:
        return brightness(img, 0.1, 0)
    elif act == 9:
        return brightness(img, 0.1, 1)
    elif act == 10:
        return brightness(img, 0.1, 2)
    elif act == 11:
        return brightness(img, -0.1, 0)
    elif act == 12:
        return brightness(img, -0.1, 1)
    elif act == 13:
        return brightness(img, -0.1, 2)
    elif act == 14:
        return brightness(img, 0.1)
    elif act == 15:
        return brightness(img, -0.1)
    elif act == 16:
        return contrast(img, 0.8, 0)
    elif act == 17:
        return contrast(img, 0.8, 1)
    elif act == 18:
        return contrast(img, 0.8, 2)
    elif act == 19:
        return contrast(img, 2, 0)
    elif act == 20:
        return contrast(img, 2, 1)
    elif act == 21:
        return contrast(img, 2, 2)
    elif act == 22:
        return contrast(img, 0.8)
    elif act == 23:
        return contrast(img, 2)
    elif act == 24:
        return saturation(img, 0.5)
    elif act == 25:
        return saturation(img, 2)
    elif act == 26:
        return hue(img, 0.05)
    elif act == 27:
        return hue(img, -0.05)
    elif act == 28:
        return Gray_World(img)
    elif act == 29:
        return MaxRGB(img)
    elif act == 30:
        return apply_filter(img, ImageFilter.MedianFilter)
    elif act == 31:
        return apply_filter(img, ImageFilter.SHARPEN)
    elif act == 32:
        return apply_filter(img, ImageFilter.GaussianBlur)
    elif act == 33:
        return apply_filter(img, ImageFilter.EDGE_ENHANCE)
    elif act == 34:
        return apply_filter(img, ImageFilter.DETAIL)
    elif act == 35:
        return apply_filter(img, ImageFilter.SMOOTH)
    elif act == 36:
        return img


def gamma_corr(image, gamma, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = mod[:, channel, :, :] ** gamma
    else:
        mod = mod ** gamma
    return mod


def brightness(image, bright, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
    else:
        mod = torch.clamp(mod + bright, 0, 1)
    return mod


def apply_filter(image, filter):
    mod = image.clone()
    mod = (transforms.ToPILImage()(mod.squeeze(0)))
    mod = transforms.ToTensor()(mod.filter(filter))
    return mod.unsqueeze(0)


def contrast(image, alpha, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(
            torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])), 0,
            1)
    else:
        mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
    return mod
