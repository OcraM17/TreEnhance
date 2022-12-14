import torch


def PSNR(img, gt):
    mseL = torch.nn.MSELoss()
    mse = mseL(img, gt)
    if mse != 0:
        print(20 * torch.log10(1 / torch.sqrt(mse)))
        return 20 * torch.log10(1 / torch.sqrt(mse))
    return 20 * torch.log10(1 / torch.sqrt(torch.tensor(1e-9)))
