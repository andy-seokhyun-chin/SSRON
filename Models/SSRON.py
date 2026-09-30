'''
The proposed SSRON architecture.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=6): # number of wavelengths?
        super().__init__()
        self.freqs = 2 ** torch.arange(num_frequencies)

    def forward(self, x):
        # x: (..., in_dim)
        out = [x]
        for f in self.freqs:
            out.append(torch.sin(f * x))
            out.append(torch.cos(f * x))
        return torch.cat(out, dim=-1)
class SpectralSpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Spatial branch (3×3)
        self.spa1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.spa2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Spectral branch (1×1)
        self.spe1 = nn.Conv2d(channels, channels, 1)
        self.spe2 = nn.Conv2d(channels, channels, 1)

        # Fusion
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 3×3 spatial path
        spa = self.relu(self.spa1(x))
        spa = self.spa2(spa)

        # 1×1 spectral path
        spe = self.relu(self.spe1(x))
        spe = self.spe2(spe)

        # Concatenate + fuse
        out = torch.cat([spa, spe], dim=1)
        out = self.fuse(out)
        return self.relu(out + x)
class SpectralSpatialNet(nn.Module):
    def __init__(self, in_channels, nBlocks, latent_dim, NNN=256):
        super().__init__()

        # Initial 1×1 projection
        self.spe1 = nn.Conv2d(in_channels, NNN, 1)

        # Residual spectral–spatial blocks
        self.blocks = nn.ModuleList([SpectralSpatialBlock(NNN) for _ in range(nBlocks)])

        # Output
        self.out_conv = nn.Conv2d(NNN, latent_dim, 1)
        
    def forward(self, x):
        x = self.spe1(x)
        for b in self.blocks:
            x=b(x)
        x = self.out_conv(x)
        return x
class TrunkMLP(nn.Module):
    def __init__(self, latent_dim, n_freq):
        super().__init__()

        self.ff = FourierFeatures(in_dim=3, num_frequencies=n_freq)

        ff_dim = 3 * (1 + 2 * n_freq)

        self.net = nn.Sequential(
            nn.Linear(ff_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, coords):
        # coords: (N, 3) where (x, y, λ)
        x = self.ff(coords)
        return self.net(x)  # (N, K)
class SSRON(nn.Module):
    def __init__(self, msi_channels, latent_dim, n_freq):
        super().__init__()

        self.branch = SpectralSpatialNet(msi_channels, 8, latent_dim=latent_dim)
        self.trunk = TrunkMLP(latent_dim, n_freq)

    def forward(self, msi, coords):
        """
        msi:    (B, C_msi, H, W)
        coords: (B, N, 3) -> (x, y, λ) normalized to [0,1]

        Returns:
            hsi_values: (B, N)
        """
        B, _, H, W = msi.shape
        _, N,_ = coords.shape

        # Branch: spatial latent field
        branch_latent = self.branch(msi)  # (B, K, H, W)
        # Trunk: coordinate latent
        coords_flat = coords.view(B * N, 3)
        trunk_latent = self.trunk(coords_flat)  # (B*N, K)
        trunk_latent = trunk_latent.view(B, N, -1)  # (B, N, K)

        # Convert normalized coords to pixel indices
        x = coords[..., 0]
        y = coords[..., 1]

        ix = (x * (W - 1)).long().clamp(0, W - 1)
        iy = (y * (H - 1)).long().clamp(0, H - 1)

        # Gather branch features per batch element
        # Output shape: (B, N, K)
        b_feat = torch.stack(
            [branch_latent[b, :, iy[b], ix[b]] for b in range(B)],
            dim=0
        ).permute(0, 2, 1)

        # Inner product
        hsi = torch.einsum("bnk,bnk->bn", b_feat, trunk_latent)

        return hsi
