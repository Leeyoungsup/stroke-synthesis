import torch

def edm2_loss(model, x, sigma, class_labels=None, cfg_drop_prob=0.1, sigma_data=0.5):
    """
    EDM2 SNR-weighted denoising loss with learned log-variance.
    """
    noise = torch.randn_like(x)
    noised_x = x + sigma[:, None, None, None] * noise

    # Classifier-Free Guidance dropout
    if class_labels is not None and torch.rand(1).item() < cfg_drop_prob:
        class_labels = None

    # forward with logvar prediction
    denoised, logvar = model(noised_x, sigma, class_labels, return_logvar=True)

    # SNR-based weight (same as EDM2 paper)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    weight = weight[:, None, None, None].expand_as(x)  # shape: [B, 1, 1, 1]
    # Final loss
    loss = (weight / logvar.exp()) * ((denoised - noise) ** 2) + logvar
    return loss.mean()
