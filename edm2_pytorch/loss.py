import torch

def edm2_loss(model, x, sigma, class_labels=None, cfg_drop_prob=0.1, sigma_data=0.5):
    """
    EDM2 SNR-weighted denoising loss (no learned log-variance, fixed sigma_data)
    based on NVlabs/edm2 official implementation.
    """
    noise = torch.randn_like(x)
    noised_x = x + sigma[:, None, None, None] * noise

    # Classifier-Free Guidance dropout
    if class_labels is not None and torch.rand(1).item() < cfg_drop_prob:
        class_labels = None

    # forward pass
    denoised = model(noised_x, sigma, class_labels)

    # SNR-based weight (same as EDM2 paper)
    snr = (sigma_data / sigma) ** 2
    weight = snr / (snr + 1)
    weight = weight[:, None, None, None].expand_as(x)

    # Primary loss
    mse_loss = weight * ((denoised - noise) ** 2)
    loss = mse_loss.mean()

    # ✨ Stabilization term to discourage fixed patterns
    loss += 1e-4 * (denoised ** 2).mean()

    return loss
