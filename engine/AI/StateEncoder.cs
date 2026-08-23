using System;
using System.Collections.Generic;

// ====================================================================
// 神经网络输入编码器（训练第一步）。
//
// 把 Gamestate 编码成 22 通道 × 14 行 × 11 列的输入特征。
// 采用固定红方视角（不翻转），"当前行动方"通道标记轮到谁走。
//
// 通道布局（与 E:\chess\AI\网络.txt 及后续约定一致）：
//   0-6:   红方 7 类棋子（車马象士将炮兵，one-hot 位置）
//   7-13:  黑方 7 类棋子（同上顺序）
//   14:    升级=基础   (upgradeLevel == 0)
//   15:    升级=1级效果 (upgradeLevel == 1 或 3)
//   16:    升级=2级效果 (upgradeLevel == 2 或 3)
//   17:    冻结剩余回合 (frozenTurns，0=未冻结)
//   18:    墙           (0/1)
//   19:    墙剩余回合   (wallDuration，0=无墙)
//   20:    狙击冷却     (sniperCooldown，0=可狙击/非狙击)
//   21:    当前行动方   (红=1 黑=0，常量平面)
//
// 张量形状：(Channels, Height, Width)，展平顺序 channel→row→col，
// 与 PyTorch (Batch, Channel, Height, Width) 的内存布局一致。
// ====================================================================
public static class StateEncoder
{
    // ── 张量形状 ──
    public const int Channels = 22;
    public const int Height = 14;              // 行（y 方向）
    public const int Width = 11;               // 列（x 方向）
    public const int PlaneSize = Height * Width;   // 154
    public const int FeatureSize = Channels * PlaneSize; // 3388

    // ── 棋盘逻辑坐标范围（领域展开后的最大范围）──
    public const int MinX = -1;
    public const int MaxX = 9;
    public const int MinY = -2;
    public const int MaxY = 11;

    // ── 动作空间维度（供后续动作编码参考）──
    public const int MoveActionSize = 23716;   // 154 × 154（from × to）
    public const int SniperActionSize = 616;   // 154 × 4（from × 方向）
    public const int LotteryActionSize = 1; // 抽奖动作本身
    public const int TotalActionSize = MoveActionSize + SniperActionSize + LotteryActionSize; // 24333

    /// <summary>
    /// 把 Gamestate 编码成 22×14×11 的 float 特征（固定红方视角，不翻转）。
    /// 返回长度为 FeatureSize 的一维数组，布局 channel→row→col。
    /// </summary>
    public static float[] Encode(Gamestate state)
    {
        float[] f = new float[FeatureSize];

        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty) continue;

                int col = x - MinX;             // x + 1，范围 0~10
                int row = y - MinY;             // y + 2，范围 0~13
                int pos = row * Width + col;

                // ── 墙：单独处理，不进棋子类型/升级通道 ──
                if (p.type == PieceType.Wall)
                {
                    f[18 * PlaneSize + pos] = 1f;
                    f[19 * PlaneSize + pos] = ((Wall)p).wallDuration;
                    continue;
                }

                // ── 1. 棋子类型通道（0-13）──
                int typeIdx = (int)p.type - 1;  // Rook=1→0 ... Pawn=7→6
                int teamBase = (p.thisTeam == 1) ? 0 : 7;
                f[(teamBase + typeIdx) * PlaneSize + pos] = 1f;

                // ── 2. 升级等级通道（14-16）──
                if (p.upgradeLevel == 0)
                    f[14 * PlaneSize + pos] = 1f;
                if (p.upgradeLevel == 1 || p.upgradeLevel == 3)
                    f[15 * PlaneSize + pos] = 1f;
                if (p.upgradeLevel == 2 || p.upgradeLevel == 3)
                    f[16 * PlaneSize + pos] = 1f;

                // ── 3. 冻结剩余回合（17）──
                f[17 * PlaneSize + pos] = p.frozenTurns;

                // ── 4. 狙击冷却（20）──
                if (p is Pawn pawn)
                    f[20 * PlaneSize + pos] = pawn.sniperCooldown;
            }
        }

        // ── 5. 当前行动方（21）：整平面填常量 ──
        float teamFlag = (state.currentTeam == 1) ? 1f : 0f;
        for (int i = 0; i < PlaneSize; i++)
            f[21 * PlaneSize + i] = teamFlag;

        return f;
    }

    // ====================================================================
    //  墓地向量编码（18 维）
    //
    //  18 维 = 18 种（棋子类型 + 升级等级）的阵亡数量（红黑合并）。
    //  顺序与 ActionEncoder.ReviveTypeLevels 一致（不含将，因将死终局）。
    // ====================================================================
    public const int GraveyardSize = 18;

    public static float[] EncodeGraveyard(Gamestate state)
    {
        float[] grave = new float[GraveyardSize];
        CountGraveyard(state.redGraveyard, grave);
        CountGraveyard(state.blackGraveyard, grave);
        return grave;
    }

    private static void CountGraveyard(List<Piece> graveyard, float[] grave)
    {
        foreach (Piece p in graveyard)
        {
            int idx = TypeLevelIndex(p.type, p.upgradeLevel);
            if (idx >= 0 && idx < GraveyardSize)
                grave[idx]++;
        }
    }

    /// <summary>类型+等级 → 18 维墓地索引（与 ActionEncoder.ReviveTypeLevels 顺序一致）</summary>
    private static int TypeLevelIndex(PieceType type, int level)
    {
        switch (type)
        {
            case PieceType.Rook: return level == 0 ? 0 : (level == 1 ? 1 : -1);
            case PieceType.Knight: return level == 0 ? 2 : (level == 1 ? 3 : -1);
            case PieceType.Bishop: return level switch { 0 => 4, 1 => 5, 2 => 6, 3 => 7, _ => -1 };
            case PieceType.Guard: return level == 0 ? 8 : (level == 1 ? 9 : -1);
            case PieceType.Cannon: return level switch { 0 => 10, 1 => 11, 2 => 12, 3 => 13, _ => -1 };
            case PieceType.Pawn: return level switch { 0 => 14, 1 => 15, 2 => 16, 3 => 17, _ => -1 };
            default: return -1; // 将/墙不进墓地
        }
    }
}
