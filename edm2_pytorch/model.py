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


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.activation = SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.emb_proj = nn.Linear(embedding_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.activation(self.norm1(x)))
        h += self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.activation(self.norm2(h)))
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
class UNetEDM2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, class_embed_dim=128, num_classes=2):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, class_embed_dim),
            SiLU(),
            nn.Linear(class_embed_dim, class_embed_dim)
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, class_embed_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, class_embed_dim)
        self.downsample1 = Downsample(base_channels * 2)

        self.middle = ResidualBlock(base_channels * 2, base_channels * 2, class_embed_dim)

        self.upsample1 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, class_embed_dim)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, class_embed_dim)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            SiLU(),
            nn.Conv2d(base_channels, out_channels * 2, 3, padding=1)  # ⭐ logvar 포함
        )

    def forward(self, x, sigma, class_labels=None, return_logvar=False):
        sigma = sigma.view(-1, 1)
        emb = self.time_embedding(sigma)

        if class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb

        x = self.input_proj(x)

        x1 = self.down1(x, emb)
        x2 = self.down2(x1, emb)
        x3 = self.downsample1(x2)

        x4 = self.middle(x3, emb)

        x5 = self.upsample1(x4)
        x5 = torch.cat([x5, x2], dim=1)
        x6 = self.up2(x5, emb)
        x6 = torch.cat([x6, x1], dim=1)
        x7 = self.up1(x6, emb)

        out = self.output_proj(x7)  # [B, 2*C, H, W]
        denoised, logvar = torch.chunk(out, 2, dim=1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0) 
        if return_logvar:
            return denoised, logvar
        else:
            return denoised
# --- EMA wrapper ---
class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
