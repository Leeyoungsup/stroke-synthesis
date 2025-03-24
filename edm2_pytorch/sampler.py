import torch

@torch.no_grad()
def edm2_sample(model, shape, sigmas, class_label=None, cfg_scale=3.0, device='cuda'):
    """
    Heun-style sampler for EDM2
    """
    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(len(sigmas) - 1):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_tensor = torch.full((shape[0],), sigma_curr, device=device)

        # Denoise with and without conditioning (for CFG)
        denoised_cond = model(x, sigma_tensor, class_label)
        if class_label is not None:
            denoised_uncond = model(x, sigma_tensor, None)
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond

        # Euler step
        d = (x - denoised) / sigma_curr
        x_euler = x + (sigma_next - sigma_curr) * d

        if i == len(sigmas) - 2:
            x = x_euler
        else:
            # Second denoising for Heun update
            sigma_next_tensor = torch.full((shape[0],), sigma_next, device=device)
            denoised_next_cond = model(x_euler, sigma_next_tensor, class_label)
            if class_label is not None:
                denoised_next_uncond = model(x_euler, sigma_next_tensor, None)
                denoised_next = denoised_next_uncond + cfg_scale * (denoised_next_cond - denoised_next_uncond)
            else:
                denoised_next = denoised_next_cond

            d_prime = (x_euler - denoised_next) / sigma_next
            x = x + (sigma_next - sigma_curr) * 0.5 * (d + d_prime)

    return x
