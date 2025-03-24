import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h += self.emb_proj(emb).view(emb.size(0), -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, cond_dim=10):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.cond_proj = nn.Linear(cond_dim, 128)

        self.down1 = ResBlock(in_ch, 64, 128)
        self.down2 = ResBlock(64, 128, 128)
        self.downsample = Downsample(128)

        self.mid1 = ResBlock(128, 128, 128)
        self.mid2 = ResBlock(128, 128, 128)

        self.upsample = Upsample(128)
        self.up1 = ResBlock(128, 64, 128)
        self.up2 = ResBlock(64, 64, 128)

        self.out_norm = nn.GroupNorm(1, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 3, padding=1)

    def get_condition_embedding(self, labels, num_classes=10):
        return F.one_hot(labels.long(), num_classes=num_classes).float()

    def forward(self, x, sigma, cond_emb=None):
        emb = self.emb(torch.log(sigma + 1e-7).view(-1, 1))
        if cond_emb is not None:
            emb += self.cond_proj(cond_emb)

        x = self.down1(x, emb)
        x = self.down2(x, emb)
        x = self.downsample(x)
        x = self.mid1(x, emb)
        x = self.mid2(x, emb)
        x = self.upsample(x)
        x = self.up1(x, emb)
        x = self.up2(x, emb)
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

class EDM2Wrapper(nn.Module):
    def __init__(self, unet_model, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.model = unet_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, class_labels=None):
        if class_labels is not None:
            cemb = self.model.get_condition_embedding(class_labels)
        else:
            cemb = None
        return self.model(x, sigma, cemb)

    def get_condition_embedding(self, labels, num_classes=10):
        # Classifier-Free Guidance를 위해 -1이면 zero embedding 사용
        labels = labels.clone()
        mask = labels == -1
        labels[mask] = 0  # 일단 0으로 설정해 one_hot에 넣음

        one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
        one_hot[mask] = 0.0  # CFG 마스킹 위치는 all-zero 벡터로 만듦
        return one_hot
    
    @torch.no_grad()
    def sample(self, shape, num_steps=18, rho=7, S_churn=0, S_noise=1.0, guidance_weight=1.5, class_labels=None):
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device) * self.sigma_max

        def get_sigmas():
            ramp = torch.linspace(0, 1, num_steps, device=device)
            return (self.sigma_max ** (1 / rho) + ramp * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho

        sigmas = get_sigmas()

        for i in range(num_steps - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            gamma = min(S_churn / num_steps, 2.0) if self.sigma_min <= sigma <= self.sigma_max else 0.0
            eps = torch.randn_like(x) * S_noise
            sigma_hat = sigma + gamma * sigma
            x_hat = x + eps * (sigma_hat**2 - sigma**2).sqrt()

            sigma_tensor = sigma_hat.expand(shape[0], 1, 1, 1)

            # CFG: 조건/비조건 score 모두 추정
            if class_labels is not None:
                score_cond = self.forward(x_hat, sigma_tensor, class_labels)
                score_uncond = self.forward(x_hat, sigma_tensor, None)
                score = score_uncond + guidance_weight * (score_cond - score_uncond)
            else:
                score = self.forward(x_hat, sigma_tensor, None)

            d = -score * (sigma_hat - sigma_next)
            x = x_hat + d

        return x.clamp(-1, 1)

# -------------------------------------------------------------
# EDM2 Loss Function
# -------------------------------------------------------------
def edm2_loss(model, x0, class_labels=None, sigma_data=0.5):
    device = x0.device
    B = x0.size(0)

    sigma = torch.randn(B, device=device).mul(1.0).exp()
    sigma = sigma.clamp(model.sigma_min, model.sigma_max).view(-1, 1, 1, 1)

    noise = torch.randn_like(x0)
    xt = x0 + sigma * noise

    if class_labels is not None:
        if len(class_labels.shape) == 0:
            class_labels = class_labels.unsqueeze(0)
        cond_emb = model.get_condition_embedding(class_labels)
    else:
        cond_emb = None

    pred = model.model(xt, sigma, cond_emb)

    target = -noise
    weight = ((sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data)) ** 2
    loss = weight * (pred - target) ** 2
    return loss.mean()

# -------------------------------------------------------------
# EDM2 Sampling Function (Euler + Stochastic Noise)
# -------------------------------------------------------------
@torch.no_grad()
def edm2_sample(model, shape, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_noise=1.0, class_labels=None):
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device) * sigma_max

    def get_sigmas():
        ramp = torch.linspace(0, 1, num_steps)
        return (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    sigmas = get_sigmas().to(device)

    for i in range(num_steps - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = min(S_churn / num_steps, 2.0) if sigma >= sigma_min and sigma <= sigma_max else 0.0
        eps = torch.randn_like(x) * S_noise
        sigma_hat = sigma + gamma * sigma
        x_hat = x + eps * (sigma_hat**2 - sigma**2).sqrt()

        score = model(x_hat, sigma_hat.expand(shape[0], 1, 1, 1), class_labels)
        d = -score * (sigma_hat - sigma_next)
        x = x_hat + d

    return x.clamp(-1, 1)
