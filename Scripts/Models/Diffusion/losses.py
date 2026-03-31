import torch.nn.functional as F

def diffusion_noise_mse(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)
