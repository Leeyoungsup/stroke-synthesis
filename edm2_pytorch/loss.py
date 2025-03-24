import torch

def edm2_loss(model, x, sigma, class_labels=None, cfg_drop_prob=0.1):
    """
    SNR-weighted denoising loss for EDM2
    """
    noise = torch.randn_like(x)
    noised_x = x + sigma[:, None, None, None] * noise

    # Classifier-Free Guidance dropout
    if class_labels is not None and torch.rand(1).item() < cfg_drop_prob:
        class_labels = None

    denoised = model(noised_x, sigma, class_labels)

    weight = 1 / (sigma ** 2)
    loss = weight[:, None, None, None] * (denoised - noise).pow(2)
    return loss.mean()
