import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Time embedding ---
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    exponent = -torch.arange(half_dim, dtype=torch.float32) / half_dim
    exponent = 10000 ** exponent
    emb = timesteps[:, None] * exponent[None, :].to(timesteps.device)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

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
        return self.down(x), x  # downsampled and skip

# --- Up block with explicit skip channels ---
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
    def __init__(self, in_ch=1, base=64, cond_dim=256, num_classes=2):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.label_embed = nn.Embedding(num_classes, cond_dim)

        self.init_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = DownBlock(base, base, cond_dim)
        self.down2 = DownBlock(base, base * 2, cond_dim, use_attn=True)
        self.down3 = DownBlock(base * 2, base * 4, cond_dim, use_attn=True)

        self.bot = ResBlock(base * 4, base * 4, cond_dim)

        self.up3 = UpBlock(in_ch=base * 4, skip_ch=base * 4, out_ch=base * 2, cond_dim=cond_dim, use_attn=True)
        self.up2 = UpBlock(in_ch=base * 2, skip_ch=base * 2, out_ch=base, cond_dim=cond_dim, use_attn=True)
        self.up1 = UpBlock(in_ch=base, skip_ch=base, out_ch=base, cond_dim=cond_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1)
        )

    def forward(self, x, sigma, y=None):
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

        return self.final(x)
