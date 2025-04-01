import torch

def edm2_loss(model, x, sigma, class_labels=None, cfg_drop=0.1, sigma_data=0.5):
    noise = torch.randn_like(x)
    noised = x + sigma[:, None, None, None] * noise
    if class_labels is not None and torch.rand(1).item() < cfg_drop:
        class_labels = None
    denoised = model(noised, sigma, class_labels)
    snr = (sigma_data / sigma) ** 2
    weight = snr / (snr + 1)
    # loss = (weight[:, None, None, None] * (denoised - noise) ** 2).mean()
    loss = ((denoised - noise) ** 2).mean()
    return loss

