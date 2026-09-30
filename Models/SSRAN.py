'''
SSRAN architecture, from https://github.com/spectralpublic/SSRAN, implemented in pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)               # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)                   # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y
class SSRAN(nn.Module):
    def __init__(self, in_channels=13, out_channels=285, NNN=256, kernel_size=5):
        super().__init__()

        # Initial feature extraction
        self.spe1 = nn.Conv2d(in_channels, NNN, kernel_size=1)

        # ---------- Block 1 ----------
        self.spe2_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)
        self.spe3_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)

        self.spe2_1 = nn.Conv2d(NNN, NNN, kernel_size=1)
        self.spe3_1 = nn.Conv2d(NNN, NNN, kernel_size=1)

        self.spe3_concat = nn.Conv2d(2 * NNN, NNN, kernel_size=1)
        self.eca1 = ECA(NNN, k_size=kernel_size)

        # ---------- Block 2 ----------
        self.spe4_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)
        self.spe5_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)

        self.spe4_1 = nn.Conv2d(NNN, NNN, kernel_size=1)
        self.spe5_1 = nn.Conv2d(NNN, NNN, kernel_size=1)

        self.spe5_concat = nn.Conv2d(2 * NNN, NNN, kernel_size=1)
        self.eca2 = ECA(NNN, k_size=kernel_size)

        # ---------- Block 3 ----------
        self.spe6_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)
        self.spe7_3 = nn.Conv2d(NNN, NNN, kernel_size=3, padding=1)

        self.spe6_1 = nn.Conv2d(NNN, NNN, kernel_size=1)
        self.spe7_1 = nn.Conv2d(NNN, NNN, kernel_size=1)

        self.spe7_concat = nn.Conv2d(2 * NNN, NNN, kernel_size=1)
        self.eca3 = ECA(NNN, k_size=kernel_size)

        # Output
        self.out_conv = nn.Conv2d(NNN, out_channels, kernel_size=1)
        self.eca_out = ECA(out_channels, k_size=kernel_size)

    def forward(self, x):
        # Initial
        spe1 = F.relu(self.spe1(x))

        # ---------- Block 1 ----------
        spe3_3 = F.relu(self.spe3_3(F.relu(self.spe2_3(spe1))))
        spe3_1 = F.relu(self.spe3_1(F.relu(self.spe2_1(spe1))))
        spe3 = torch.cat([spe3_1, spe3_3], dim=1)
        spe3 = F.relu(self.spe3_concat(spe3))
        spe3 = self.eca1(spe3)
        spe3 = F.relu(spe3 + spe1)

        # ---------- Block 2 ----------
        spe5_3 = F.relu(self.spe5_3(F.relu(self.spe4_3(spe3))))
        spe5_1 = F.relu(self.spe5_1(F.relu(self.spe4_1(spe3))))
        spe5 = torch.cat([spe5_3, spe5_1], dim=1)
        spe5 = F.relu(self.spe5_concat(spe5))
        spe5 = self.eca2(spe5)
        spe5 = F.relu(spe5 + spe3)

        # ---------- Block 3 ----------
        spe7_3 = F.relu(self.spe7_3(F.relu(self.spe6_3(spe5))))
        spe7_1 = F.relu(self.spe7_1(F.relu(self.spe6_1(spe5))))
        spe7 = torch.cat([spe7_3, spe7_1], dim=1)
        spe7 = F.relu(self.spe7_concat(spe7))
        spe7 = self.eca3(spe7)
        spe7 = F.relu(spe7 + spe5)

        # Output
        out = self.out_conv(spe7)
        out = self.eca_out(out)
        return out
