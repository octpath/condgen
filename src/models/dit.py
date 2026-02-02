"""DiT (Diffusion Transformer) with AdaLN-Zero for Age Conditioning.

Vision Transformer ベースの Diffusion モデル。AdaLN-Zero により、
Time + Age の連続条件を各 Transformer Block に強力に注入する。
本番では ModernBERT への移行を想定した布石。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DiT_Tiny", "DiT_ComplexCond", "GaussianFourierProjection"]


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Projection for continuous scalar (e.g. age).

    B05 の Neural Fields の概念に基づく。学習しないランダム重みで
    スカラーを高次元の sin/cos 特徴に写し、補間を滑らかにする。
    """

    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        half_dim = embed_dim // 2
        # 学習しないランダムな重み (requires_grad=False)
        self.register_buffer("W", torch.randn(half_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) -> (B, embed_dim)"""
        x_proj = x[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """DDPM 風の正弦波タイムステップ埋め込み。(B,) -> (B, embedding_dim)."""
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device
    ) / half_dim
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb


class PatchEmbed(nn.Module):
    """画像を Patch に分割し、Embedding する。"""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, embed_dim)"""
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# -----------------------------------------------------------------------------
# AdaLN-Zero: Adaptive Layer Normalization with Zero Initialization
# -----------------------------------------------------------------------------


class AdaLNZero(nn.Module):
    """AdaLN-Zero: 条件埋め込み (Time + Age) から LayerNorm の Scale/Shift を回帰。
    
    DiT 論文の核心部分。条件なしではブロック内の演算がほぼゼロになるように、
    Scale/Shift をゼロ初期化する。
    """

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        """
        Args:
            cond_dim: 条件埋め込みの次元（Time + Age の合計）。
            hidden_dim: Transformer の隠れ層次元。
        """
        super().__init__()
        # 条件から 6 つのパラメータを回帰: shift_1, scale_1, gate_1, shift_2, scale_2, gate_2
        # (Attention 用と MLP 用で各 3 つずつ)
        self.linear = nn.Linear(cond_dim, hidden_dim * 6)
        # ゼロ初期化: 学習開始時は条件の影響ゼロ
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """条件から Scale/Shift を計算。
        
        Args:
            cond: (B, cond_dim) 条件埋め込み。
        
        Returns:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            各 (B, hidden_dim)。
        """
        params = self.linear(cond)  # (B, hidden_dim * 6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(
            6, dim=-1
        )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


# -----------------------------------------------------------------------------
# DiT Block (Transformer Block with AdaLN-Zero)
# -----------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Transformer Block with AdaLN-Zero conditioning.
    
    標準的な Self-Attention + MLP だが、AdaLN-Zero で条件付けする。
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN = AdaLNZero(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) パッチトークン列。
            cond: (B, cond_dim) 条件埋め込み (Time + Age)。
        
        Returns:
            (B, N, hidden_dim) 更新されたトークン列。
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            cond
        )

        # Self-Attention with AdaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# -----------------------------------------------------------------------------
# Final Layer (Linear Decoder)
# -----------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """最終層: パッチ埋め込みを元の画像パッチ次元に射影。AdaLN で条件付け。"""

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        out_channels: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)
            cond: (B, cond_dim)
        
        Returns:
            (B, N, patch_size^2 * out_channels)
        """
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# -----------------------------------------------------------------------------
# DiT_Tiny Model
# -----------------------------------------------------------------------------


class DiT_Tiny(nn.Module):
    """DiT-Tiny: 軽量な Diffusion Transformer with AdaLN-Zero.
    
    - Patch Embedding → Transformer Blocks (AdaLN-Zero) → Final Layer
    - Time + Age による強力な連続条件付け
    - ModernBERT への移行を想定した設計
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
        age_embed_dim: int = 128,
    ) -> None:
        """
        Args:
            img_size: 入力画像サイズ (64)。
            patch_size: パッチサイズ (4 or 8)。
            in_channels: 入力チャンネル数 (3: RGB)。
            hidden_dim: Transformer 隠れ層次元 (384)。
            depth: Transformer レイヤー数 (6)。
            num_heads: Attention ヘッド数 (6)。
            mlp_ratio: MLP 拡張率 (4.0)。
            time_embed_dim: Time Embedding 次元 (128)。
            age_embed_dim: Age Embedding 次元 (128)。
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        # 2次元位置埋め込み（必須）: パッチの空間位置を表現。これがないと画像を構成できない。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time + Age Embedding
        self.time_embed_dim = time_embed_dim
        self.age_embed_dim = age_embed_dim
        cond_dim = time_embed_dim + age_embed_dim
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        # Age: Gaussian Fourier Projection -> MLP (B05 VRM / Fourier Feature)
        self.age_fourier = GaussianFourierProjection(age_embed_dim, scale=30.0)
        self.age_mlp = nn.Sequential(
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )

        # Transformer Blocks with AdaLN-Zero
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim)
                for _ in range(depth)
            ]
        )

        # Final Layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels, cond_dim)
        
        # 畳み込みデコーダー（パッチアーティファクト除去）
        self.final_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """パッチ列を画像に復元。
        
        Args:
            x: (B, N, patch_size^2 * C)
        
        Returns:
            (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, h, p, w, p)
        x = x.reshape(B, self.out_channels, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        """ノイズ予測を行う。
        
        Args:
            x: (B, C, H, W) ノイズ付き画像。
            timesteps: (B,) タイムステップ。
            age: (B,) 年齢（正規化済み [0,1]）。
        
        Returns:
            (B, C, H, W) ノイズ予測。
        """
        B = x.shape[0]

        # Patch Embedding + 2次元位置埋め込み（必須）
        x = self.patch_embed(x)  # (B, N, hidden_dim)
        x = x + self.pos_embed  # 空間位置情報を付与

        # Time Embedding
        t_emb = sinusoidal_timestep_embedding(
            timesteps, self.time_embed_dim
        )  # (B, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_embed_dim)

        # Age Embedding: Fourier Feature -> MLP (補間滑らか、斜めアーティファクト軽減)
        a_emb = self.age_fourier(age)  # (B, age_embed_dim)
        a_emb = self.age_mlp(a_emb)  # (B, age_embed_dim)

        # Condition: Time + Age
        cond = torch.cat([t_emb, a_emb], dim=-1)  # (B, time_embed_dim + age_embed_dim)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final Layer
        x = self.final_layer(x, cond)  # (B, N, patch_size^2 * C)

        # Unpatchify
        x = self.unpatchify(x)  # (B, C, H, W)
        
        # 畳み込みデコーダー（パッチの継ぎ目を滑らかに）
        x = self.final_decoder(x)  # (B, C, H, W)

        return x


# -----------------------------------------------------------------------------
# DiT_ComplexCond: 年齢（連続）+ 性別（カテゴリ）の複合条件付け
# -----------------------------------------------------------------------------


class DiT_ComplexCond(nn.Module):
    """DiT with 複合条件: 年齢 (連続) + 性別 (カテゴリ)。

    - Age: GaussianFourierProjection -> MLP -> v_age
    - Gender: nn.Embedding(2, embedding_dim) -> v_gender
    - Fusion: Concat(v_age, v_gender) -> Linear -> v_fused
    - Injection: v_fused + t_emb -> AdaLN (Scale/Shift)
    - pos_embed と Final Conv Decoder は必須として維持。
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
        age_embed_dim: int = 64,
        gender_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed_dim = time_embed_dim
        self.cond_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.age_fourier = GaussianFourierProjection(age_embed_dim, scale=30.0)
        self.age_mlp = nn.Sequential(
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )
        self.gender_embed = nn.Embedding(2, gender_embed_dim)
        self.fusion_linear = nn.Linear(age_embed_dim + gender_embed_dim, time_embed_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio, self.cond_dim)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels, self.cond_dim)
        self.final_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
    ) -> torch.Tensor:
        """ノイズ予測。age: (B,) 正規化 [0,1], gender: (B,) long 0=Male, 1=Female。"""
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed

        t_emb = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        a_emb = self.age_fourier(age)
        a_emb = self.age_mlp(a_emb)
        g_emb = self.gender_embed(gender.squeeze(-1) if gender.dim() > 1 else gender)
        v_concat = torch.cat([a_emb, g_emb], dim=-1)
        v_fused = self.fusion_linear(v_concat)
        cond = v_fused + t_emb

        for block in self.blocks:
            x = block(x, cond)
        x = self.final_layer(x, cond)
        x = self.unpatchify(x)
        x = self.final_decoder(x)
        return x


# -----------------------------------------------------------------------------
# Main (動作確認)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT_Tiny(
        img_size=64,
        patch_size=8,
        in_channels=3,
        hidden_dim=384,
        depth=6,
        num_heads=6,
    ).to(device)

    B = 4
    x = torch.randn(B, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    age = torch.rand(B).to(device)  # [0, 1]

    print(f"Input shape: {x.shape}")
    print(f"Timesteps: {t}")
    print(f"Ages: {age}")

    with torch.no_grad():
        out = model(x, t, age)

    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("DiT_Tiny with AdaLN-Zero: OK")
