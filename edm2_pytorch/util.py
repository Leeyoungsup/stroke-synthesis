import torch

def get_sigmas_karras_with_p(n=40,P_mean=-0.4, P_std=1.0, device='cuda'):
    """
    EDM2 + CFG 스타일 확장:
    P_mean, P_std를 기반으로 log-normal 분포에서 샘플링한 noise level을 사용

    Returns:
        sigma: [n] shaped tensor of noise levels
    """
    # log-normal distribution에서 log_sigma 샘플링
    log_sigma = torch.randn(n, device=device) * P_std + P_mean
    sigma = torch.exp(log_sigma)

    # rho 방식으로 정렬 (선택 사항)
    sigma = torch.sort(sigma, descending=True)[0]

    return sigma