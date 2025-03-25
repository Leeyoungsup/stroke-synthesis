import torch

def get_sigmas_karras_with_p(n=40, P_mean=-0.4, P_std=1.0, rho=7.0, device='cuda'):
    """
    EDM2 스타일 샘플러 스케줄:
    - log-normal 기반으로 sigma_min, sigma_max 결정 (P_mean, P_std)
    - rho 보간 스케줄 적용

    Returns:
        sigmas: [n] shaped tensor of noise levels (정렬된 스케줄)
    """
    # 1. sigma_min, sigma_max 계산 (log-normal 분포의 0.001, 0.999 분위수 근사)
    sigma_max = torch.exp(torch.tensor(P_mean + 2 * P_std))
    sigma_min = torch.exp(torch.tensor(P_mean - 2 * P_std))

    # 2. rho-schedule 기반 보간
    t = torch.linspace(0, 1, n, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    return sigmas
