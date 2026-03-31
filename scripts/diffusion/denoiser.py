import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int):
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )
    args = timesteps[:, None].float() * freqs[None]

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, time_dim, drop_prob=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_chans)

        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_prob)

        self.time_mlp = nn.Linear(time_dim, out_chans * 2)

        if in_chans != out_chans:
            self.res_conv = nn.Conv2d(in_chans, out_chans, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.res_conv(x)

        h = self.conv1(x)
        h = self.norm1(h)

        time_emb = self.time_mlp(t_emb)
        scale, shift = time_emb.chunk(2, dim=1)
        scale = scale.view(scale.shape[0], -1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1)

        h = h * (1 + scale) + shift
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        h = self.dropout(h)

        return h + residual


class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H * W).permute(0, 2, 1)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        return x + attn_out


class UpConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, 2, stride=2, bias=False),
            nn.GroupNorm(8, out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DenoiserUNet(nn.Module):

    def __init__(
        self,
        base_chans: int = 128,
        num_pool_layers: int = 4,
        time_embed_dim: int = 128,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        self.pool = nn.AvgPool2d(2)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.down_blocks = nn.ModuleList()

        in_ch = 2
        down_channels = []

        ch = base_chans
        for _ in range(num_pool_layers):
            self.down_blocks.append(
                ConvBlock(in_ch, ch, time_embed_dim, drop_prob)
            )
            down_channels.append(ch)
            in_ch = ch
            ch *= 2


        bottleneck_ch = down_channels[-1]

        self.bottleneck = ConvBlock(
            bottleneck_ch,
            bottleneck_ch,
            time_embed_dim,
            drop_prob,
        )

        self.attn = SelfAttention2D(bottleneck_ch)

        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        reversed_channels = list(reversed(down_channels))
        in_ch = bottleneck_ch

        for skip_ch in reversed_channels:
            self.up_convs.append(UpConv(in_ch, skip_ch))
            self.up_blocks.append(
                ConvBlock(skip_ch * 2, skip_ch, time_embed_dim, drop_prob)
            )
            in_ch = skip_ch

        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x_t, masked, t):

        x = torch.cat([x_t, masked], dim=1)

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        skips = []

        for block in self.down_blocks:
            x = block(x, t_emb)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t_emb)
        x = self.attn(x)

        skips = list(reversed(skips))

        for up, block, skip in zip(self.up_convs, self.up_blocks, skips):
            x = up(x)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        return self.final_conv(x)