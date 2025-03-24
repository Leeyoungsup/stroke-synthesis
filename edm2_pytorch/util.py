import torch

def get_sigmas_karras(n=40, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cuda'):
    """
    EDM2-style noise schedule using Karras et al. method
    """
    steps = torch.linspace(0, 1, n, device=device)
    sigma = (sigma_max**(1/rho) + steps * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    return sigma
