import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
import math
import torch

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings as in EDM2 official implementation.
    Args:
        timesteps: Tensor of shape [B], containing timesteps (or log-sigma).
        embedding_dim: Dimension of the embedding vector.
    Returns:
        Tensor of shape [B, embedding_dim]
    """
    assert len(timesteps.shape) == 1  # (B,)
    half_dim = embedding_dim // 2

    # Create frequencies
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )  # [half_dim]

    # Outer product: [B, 1] * [1, half_dim] => [B, half_dim]
    args = timesteps[:, None].float() * freqs[None, :]

    # Sinusoidal embeddings
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embedding_dim]
    return emb

# --- FiLM conditioning ---
class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.film = nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

# --- Residual block ---
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.film = FiLM(cond_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

# --- Attention block ---
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / C ** 0.5, dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return x + self.proj(out)

# --- Down block ---
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, use_attn=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.AvgPool2d(2)

    def forward(self, x, cond):
        x = self.res(x, cond)
        x = self.attn(x)
        return self.down(x), x

# --- Up block ---
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, use_attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, cond)
        x = self.attn(x)
        return x

# --- Full UNet ---
class EDM2UNet(nn.Module):
    def __init__(self, in_ch=1, base=128, cond_dim=256, num_classes=2):
        super().__init__()
        ch_mult = [1, 2, 4, 8]
        chs = [base * m for m in ch_mult]

        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.label_embed = nn.Embedding(num_classes, cond_dim)

        self.init_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.down1 = DownBlock(chs[0], chs[1], cond_dim)
        self.down2 = DownBlock(chs[1], chs[2], cond_dim, use_attn=True)
        self.down3 = DownBlock(chs[2], chs[3], cond_dim, use_attn=True)

        self.bot = ResBlock(chs[3], chs[3], cond_dim)

        self.up3 = UpBlock(chs[3], chs[3], chs[2], cond_dim, use_attn=True)
        self.up2 = UpBlock(chs[2], chs[2], chs[1], cond_dim, use_attn=True)
        self.up1 = UpBlock(chs[1], chs[1], chs[0], cond_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_ch, 3, padding=1)
        )

        # logvar parameter for uncertainty-aware loss
        self.learned_logvar = nn.Parameter(torch.zeros(1))

    def forward(self, x, sigma, y=None, return_logvar=False):
        sigma = sigma.view(-1, 1)
        log_sigma = sigma.log()
        cond = get_timestep_embedding(log_sigma.squeeze(), 256).to(x.device)

        if y is not None:
            cond += self.label_embed(y)
        cond = self.time_embed(cond)

        x = self.init_conv(x)
        x1_out, x1_skip = self.down1(x, cond)
        x2_out, x2_skip = self.down2(x1_out, cond)
        x3_out, x3_skip = self.down3(x2_out, cond)

        mid = self.bot(x3_out, cond)

        x = self.up3(mid, x3_skip, cond)
        x = self.up2(x, x2_skip, cond)
        x = self.up1(x, x1_skip, cond)

        if return_logvar:
            return self.final(x), self.learned_logvar.expand_as(self.final(x))
        else:
            return self.final(x)
# --- EMA wrapper ---
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.ema_model = deepcopy(model)
        self.decay = decay
        self._disable_grad(self.ema_model)

    def _disable_grad(self, model):
        for p in model.parameters():
            p.requires_grad_(False)

    def update(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1. - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.ema_model.to(device)
        return self

# --- Sample function (EMA 적용 포함) ---
@torch.no_grad()
def sample_with_ema(model, shape, sigmas, class_label=None, cfg_scale=2.0, device='cuda'):
    model = model.ema_model
    model.eval()
    x = torch.randn(shape, device=device) * sigmas[0]
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_tensor = torch.full((shape[0],), sigma_curr, device=device)

        denoised_cond = model(x, sigma_tensor, class_label)
        if class_label is not None:
            denoised_uncond = model(x, sigma_tensor, None)
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond

        d = (x - denoised) / sigma_curr
        x_euler = x + (sigma_next - sigma_curr) * d

        if i == len(sigmas) - 2:
            x = x_euler
        else:
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
@torch.no_grad()
def euler_sampler(model, shape, sigmas, class_label=None, cfg_scale=2.0, device='cuda'):
    x = torch.randn(shape, device=device) * sigmas[0]
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_tensor = torch.full((shape[0],), sigma, device=device)

        denoised = model(x, sigma_tensor, class_label)
        if class_label is not None:
            uncond = model(x, sigma_tensor, None)
            denoised = uncond + cfg_scale * (denoised - uncond)

        d = (x - denoised) / sigma
        x = x + (sigma_next - sigma) * d
    return x
