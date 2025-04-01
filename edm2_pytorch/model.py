# EDM2 official-style refactored code

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

# --- Sinusoidal Timestep Embedding ---
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# --- SiLU Activation ---
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- Residual Block ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h += self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

# --- Downsample / Upsample ---
class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.deconv(x)

# --- UNet for EDM2 ---
class UNetEDM2(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, emb_ch=128, num_classes=2):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, emb_ch)
        self.emb_ch = emb_ch

        self.input = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, base, emb_ch)
        self.down2 = ResBlock(base, base*2, emb_ch)
        self.downsample = Down(base*2)
        self.middle = ResBlock(base*2, base*2, emb_ch)
        self.up = Up(base*2)
        self.up2 = ResBlock(base*4, base, emb_ch)
        self.up1 = ResBlock(base*2, base, emb_ch)

        self.output = nn.Sequential(
            nn.GroupNorm(8, base), SiLU(),
            nn.Conv2d(base, out_ch * 2, 3, padding=1)
        )

    def forward(self, x, sigma, class_labels=None, return_logvar=False):
        log_sigma = sigma.log()
        emb = get_timestep_embedding(log_sigma, self.emb_ch)
        if class_labels is not None:
            emb += self.class_embed(class_labels)

        x0 = self.input(x)
        x1 = self.down1(x0, emb)
        x2 = self.down2(x1, emb)
        x3 = self.downsample(x2)
        x4 = self.middle(x3, emb)
        x5 = self.up(x4)
        x5 = torch.cat([x5, x2], dim=1)
        x6 = self.up2(x5, emb)
        x6 = torch.cat([x6, x1], dim=1)
        x7 = self.up1(x6, emb)
        out = self.output(x7)
        denoised, logvar = out.chunk(2, dim=1)
        return (denoised, logvar) if return_logvar else denoised

# --- EMA Wrapper ---
class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema_model.parameters(), model.parameters()):
            e.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
