import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 双头神经网络（策略 + 价值），AlphaZero 风格。
#
# 输入：棋盘特征 (B, 22, 14, 11) + 墓地向量 (B, 18)
# 输出：策略 logits (B, 24333) + 价值 (B, 1) in [-1, 1]
#
# 动作空间索引（必须与 C# 的 StateEncoder/ActionEncoder 一致）：
#   移动 0~23715 (from×to)、狙击 23716~24331 (from×方向)、抽奖 24332~27151
#
# 展平顺序关键点（与 C# ActionEncoder 对齐）：
#   - 移动：双线性 from×to，展平 from*154 + to
#   - 狙击：permute 成 (from, dir)，展平 from*4 + dir
#   - 抽奖：18 个"选格子"槽位(各154) + 复活48，展平 slot*154 + cell，末尾接复活
# ====================================================================

BOARD_H = 14
BOARD_W = 11
INPUT_CH = 22
GRAVEYARD = 18
MOVE_SIZE = 23716        # 154 × 154
SNIPER_SIZE = 616        # 154 × 4
LOTTERY_SIZE = 1         # 抽奖动作本身
TOTAL_ACTION_SIZE = MOVE_SIZE + SNIPER_SIZE + LOTTERY_SIZE   # 24333


class ResBlock(nn.Module):
    """标准残差块：Conv3x3 + BN + ReLU + Conv3x3 + BN + 残差连接"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self, num_blocks=8, channels=128, move_emb=64):
        super().__init__()
        self.channels = channels

        # ── 输入卷积 ──
        self.conv_input = nn.Conv2d(INPUT_CH, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # ── 残差块堆叠 ──
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # ── 移动头（双线性 from × to）──
        self.move_from_conv = nn.Conv2d(channels, move_emb, 1)
        self.move_to_conv = nn.Conv2d(channels, move_emb, 1)

        # ── 狙击头 ──
        self.sniper_conv1 = nn.Conv2d(channels, 256, 1, bias=False)
        self.sniper_bn1 = nn.BatchNorm2d(256)
        self.sniper_conv2 = nn.Conv2d(256, 4, 1, bias=False)  # 4 方向

        # ── 抽奖头：选格子（卷积）+ 复活（全连接）──
        self.lottery_fc = nn.Sequential(
            nn.Linear(channels + GRAVEYARD, 256),
            nn.ReLU(),
            nn.Linear(256, LOTTERY_SIZE),
        )

        # ── 价值头 ──
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc = nn.Sequential(
            nn.Linear(BOARD_H * BOARD_W + GRAVEYARD, 256),  # 154 + 18
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, graveyard):
        # x: (B, 22, 14, 11)，graveyard: (B, 18)
        B = x.size(0)

        # ── 共享特征 ──
        shared = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            shared = block(shared)

        # ── 移动头（双线性）──
        from_emb = self.move_from_conv(shared).flatten(2)          # (B, d, 154)
        to_emb = self.move_to_conv(shared).flatten(2)              # (B, d, 154)
        move = torch.bmm(from_emb.transpose(1, 2), to_emb)         # (B, 154, 154) from×to
        move = move.reshape(B, MOVE_SIZE)                          # from*154+to

        # ── 狙击头 ──
        sniper = F.relu(self.sniper_bn1(self.sniper_conv1(shared)))
        sniper = self.sniper_conv2(sniper)                          # (B, 4, 14, 11)
        sniper = sniper.permute(0, 2, 3, 1).reshape(B, SNIPER_SIZE) # from*4+dir

        # ── 抽奖头 ──
        pooled = shared.mean(dim=(2, 3))                            # (B, channels)
        lottery = self.lottery_fc(torch.cat([pooled, graveyard], dim=1))  # (B, 1)

        # ── 价值头 ──
        v = F.relu(self.value_bn(self.value_conv(shared)))          # (B, 1, 14, 11)
        v = v.flatten(1)                                            # (B, 154)
        v = torch.tanh(self.value_fc(torch.cat([v, graveyard], dim=1)))  # (B, 1)

        # ── 拼接策略 ──
        policy = torch.cat([move, sniper, lottery], dim=1)          # (B, 24333)

        return policy, v
