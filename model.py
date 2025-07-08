#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
    
    
class KeypointNet(nn.Module):
    """
    Input : (B, 1, 96, 96)   grayscale 9696
    Output: (B, 30)         15 (x,y) key-points
    """

    def __init__(self):
        super().__init__()

        leak = 0.1  # negative-slope for LeakyReLU

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),  # "same" padding
                nn.LeakyReLU(leak, inplace=True),
                nn.BatchNorm2d(out_c),
            )

        self.features = nn.Sequential(
            # 96 × 96
            block(1,    32),
            block(32,   32),
            nn.MaxPool2d(2),          # 48 × 48

            block(32,   64),
            block(64,   64),
            nn.MaxPool2d(2),          # 24 × 24

            block(64,   96),
            block(96,   96),
            nn.MaxPool2d(2),          # 12 × 12

            block(96,  128),
            block(128, 128),
            nn.MaxPool2d(2),          # 6 × 6

            block(128, 256),
            block(256, 256),
            nn.MaxPool2d(2),          # 3 × 3

            block(256, 512),
            block(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 512 × 3 × 3 = 4608
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 30),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class KeypointNet2(nn.Module):
    """
    Input : (B, 1, 96, 96)   grayscale 9696
    Output: (B, 30)         15 (x,y) key-points
    """

    def __init__(self):
        super().__init__()

        leak = 0.1  # negative-slope for LeakyReLU

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),  # "same" padding
                nn.LeakyReLU(leak, inplace=True),
                nn.BatchNorm2d(out_c),
            )

        self.features = nn.Sequential(
            # 96 × 96
            block(1,    32),
            block(32,   32),
            nn.MaxPool2d(2),          # 48 × 48

            block(32,   64),
            block(64,   64),
            nn.MaxPool2d(2),          # 24 × 24

            block(64,   96),
            block(96,   96),
            nn.MaxPool2d(2),          # 12 × 12

            block(96,  128),
            block(128, 128),
            nn.MaxPool2d(2),          # 6 × 6

            block(128, 256),
            block(256, 256),
            nn.MaxPool2d(2),          # 3 × 3

            block(256, 512),
            block(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 512 × 3 × 3 = 4608
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 30),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
if __name__ == "__main__":
    model = KeypointNet2()
    total_params = sum(p.numel() for p in model.parameters())
    sz_mb = total_params * 4 / 1024**2
    print(f"{total_params/1e6:.2f} M parameters  (~{sz_mb:.1f} MB)")
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Simple 2×Conv residual block."""
    def __init__(self, channels, dilation=1, leak=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.LeakyReLU(leak, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)          # residual add


def conv_bn_act(in_c, out_c, leak=0.1, stride=1):
    """3×3 conv → BN → LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(leak, inplace=True),
    )


class KeypointNet3(nn.Module):
    """
    Input : (B, 1, 96, 96)
    Output: (B, 30)
    """
    def __init__(self, leak=0.1):
        super().__init__()
        # -------- Stem & stages --------
        self.stage1 = nn.Sequential(                # 96×96
            conv_bn_act(1,   32, leak),
            ResBlock(32, leak=leak),
            nn.MaxPool2d(2),                        # 48×48
        )

        self.stage2 = nn.Sequential(
            conv_bn_act(32,  64, leak),
            ResBlock(64, leak=leak),
            nn.MaxPool2d(2),                        # 24×24
        )

        self.stage3 = nn.Sequential(
            conv_bn_act(64,  128, leak),
            ResBlock(96, leak=leak),
            nn.MaxPool2d(2),                        # 12×12
        )

        self.stage4 = nn.Sequential(
            conv_bn_act(128, 256, leak),
            ResBlock(128, leak=leak),
            nn.MaxPool2d(2),                        # 6×6
        )

        # last stage with **dilated convs** for larger RF
        self.stage5 = nn.Sequential(
            conv_bn_act(256, 512, leak),
            ResBlock(512, leak=leak, dilation=2),   # 6×6, RF↑
            nn.MaxPool2d(2),                        # 3×3
            conv_bn_act(512, 512, leak),            # 3×3
        )

        # -------- Head --------
        # Global average pooling wipes the spatial dims ⇒ (B,512,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 1×1 conv acts like a fully-connected layer over channels
        self.squeeze = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leak, inplace=True),
        )

        # tiny classifier (≈ 140 k params instead of 2.5 M)
        self.fc = nn.Sequential(
            nn.Flatten(),            # (B,512)
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(leak, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 30),
        )

        # Kaiming init for convs, zeros for BN biases (PyTorch default OK)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)      # (B,512,3,3)
        x = self.gap(x)         # (B,512,1,1)
        x = self.squeeze(x)     # (B,512,1,1)
        return self.fc(x)       # (B,30)


# quick sanity check
if __name__ == "__main__":
    model = KeypointNet3()
    total_params = sum(p.numel() for p in model.parameters())
    sz_mb = total_params * 4 / 1024**2
    print(f"{total_params/1e6:.2f} M parameters  (~{sz_mb:.1f} MB)")

#%%

import torch
import torch.nn as nn

# ---------- building blocks ----------
class ResBlock(nn.Module):
    def __init__(self, c, dilation=1, leak=0.1):
        super().__init__()
        self.act = nn.LeakyReLU(leak, inplace=True)
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


def conv_bn_act(inp, out, leak=0.1, stride=1):
    """3×3 Conv → BN → LeakyReLU"""
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.LeakyReLU(leak, inplace=True),
    )

# ---------- network ----------
class KeypointNetM(nn.Module):
    """
    A mid-sized CNN for 96×96 greyscale facial-landmark regression.
    Output: 30 values (15 (x, y) pairs)
    Params: ≈ 13.2 M  (~ 50 MB fp32  /  25 MB fp16)
    """
    def __init__(self, leak: float = 0.1):
        super().__init__()
        W = [40, 80, 112, 144, 256, 512]      # channel widths

        layers, in_c = [], 1
        for w in W[:-1]:                       # stages 1-4
            layers += [
                conv_bn_act(in_c, w, leak),
                ResBlock(w, leak=leak),
                nn.MaxPool2d(2)               # 96→48→24→12→6
            ]
            in_c = w

        # last stage (6 × 6 → 3 × 3) with dilation
        layers += [
            conv_bn_act(W[-2], W[-1], leak),
            ResBlock(W[-1], dilation=2, leak=leak)
        ]
        self.features = nn.Sequential(*layers)   # → (B, 512, 3, 3)

        # classifier head (~ 3 M params)
        self.head = nn.Sequential(
            nn.Flatten(),                        # 512·3·3 = 4608
            nn.Linear(4608, 1024),
            nn.LeakyReLU(leak, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 30),
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x):          # (B,1,96,96)
        x = self.features(x)       # (B,512,3,3)
        return self.head(x)        # (B,30)

# quick size check
if __name__ == "__main__":
    model = KeypointNetM()
    p = sum(p.numel() for p in model.parameters())
    print(f"{p/1e6:.2f} M parameters  (~{p*4/1024**2:.1f} MB)")


# %%
##########################################
########### TRANSFORMER MODEL ############
##########################################
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split image into patches and embed"""

    def __init__(self, img_size: int = 96, patch_size: int = 8, in_chans: int = 1, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)                      # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2)                      # [B, embed_dim, N]
        x = x.transpose(1, 2)                 # [B, N, embed_dim]
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 dropout: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, dropout: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTFaceKeypoint(nn.Module):
    def __init__(self,
                 img_size: int = 96,
                 patch_size: int = 8,
                 in_chans: int = 1,
                 num_keypoints: int = 15,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.,
                 dropout: float = 0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_keypoints * 2)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 96, 96]
        x = self.patch_embed(x)  # [B, N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Mean pool over patch dimension
        x = x.mean(dim=1)  # [B, D]

        x = self.head(x)   # [B, 30]
        return x


if __name__ == "__main__":  # quick parameter check
    model = ViTFaceKeypoint()
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = num_params * 4 / (1024 ** 2)
    print(f"Total parameters: {num_params / 1e6:.2f}M  (~{size_mb:.1f}\u202fMB)")
    
  
# %%
