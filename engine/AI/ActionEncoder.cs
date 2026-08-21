using System;
using System.Collections.Generic;

// ====================================================================
// 动作编码器：把 GameAction 映射到策略头输出的 logits 索引（0~27151）。
//
// 索引布局（与 StateEncoder.TotalActionSize = 27152 一致）：
//   移动：0       ~ 23715   (from × to，各 154 格)
//   狙击：23716   ~ 24331   (from × 方向4)
//   抽奖：24332   ~ 27151   (抽奖后续选择，2820 槽位)
//
// 抽奖槽位布局（2820，按效果分组，详见 GetLotterySlotStart）：
//   - "选一个格子"的效果占 154 槽（棋盘 154 格）
//   - "选两个格子"的效果占 308 槽（第一个 154 + 第二个 154，顺序敏感）
//   - "复活"占 48 槽（18 种类型+等级 × 各自初始位置）
//
// 说明：抽奖后续选择是抽奖结果确定后的二级决策。编码采用固定槽位，
//       复活槽位用"当前玩家视角"的初始位置（解码时按 state.currentTeam 查坐标）。
// ====================================================================
public static class ActionEncoder
{
    // ── 格子索引 ──
    public static int CellIndex(int x, int y)
    {
        return (y - StateEncoder.MinY) * StateEncoder.Width + (x - StateEncoder.MinX);
    }

    // ── 狙击方向索引 ──
    public static int DirectionIndex(int dx, int dy)
    {
        if (dx == 0 && dy == 1) return 0;    // 上
        if (dx == 0 && dy == -1) return 1;   // 下
        if (dx == -1 && dy == 0) return 2;   // 左
        return 3;                             // 右 (1,0)
    }

    // ================================================================
    //  移动编码 / 解码
    // ================================================================
    public static int EncodeMove(int fromX, int fromY, int toX, int toY)
    {
        return CellIndex(fromX, fromY) * 154 + CellIndex(toX, toY);
    }

    public static (int fromX, int fromY, int toX, int toY) DecodeMove(int index)
    {
        int from = index / 154;
        int to = index % 154;
        int fromX = from % StateEncoder.Width + StateEncoder.MinX;
        int fromY = from / StateEncoder.Width + StateEncoder.MinY;
        int toX = to % StateEncoder.Width + StateEncoder.MinX;
        int toY = to / StateEncoder.Width + StateEncoder.MinY;
        return (fromX, fromY, toX, toY);
    }

    // ================================================================
    //  狙击编码 / 解码
    // ================================================================
    public static int EncodeSniper(int fromX, int fromY, int dx, int dy)
    {
        return StateEncoder.MoveActionSize + CellIndex(fromX, fromY) * 4 + DirectionIndex(dx, dy);
    }

    public static (int fromX, int fromY, int dx, int dy) DecodeSniper(int index)
    {
        int rel = index - StateEncoder.MoveActionSize;
        int from = rel / 4;
        int dir = rel % 4;
        int fromX = from % StateEncoder.Width + StateEncoder.MinX;
        int fromY = from / StateEncoder.Width + StateEncoder.MinY;
        (int dx, int dy) = dir switch
        {
            0 => (0, 1),
            1 => (0, -1),
            2 => (-1, 0),
            _ => (1, 0),
        };
        return (fromX, fromY, dx, dy);
    }

    // ================================================================
    //  抽奖后续选择槽位布局
    //
    //  返回每个 outcome 的槽位起始索引（相对抽奖段 0~2819）。
    //  -1 表示该 outcome 无后续选择（自动生效 / 御驾亲征 / 未中奖）。
    // ================================================================
    public static int GetLotterySlotStart(int outcome)
    {
        switch (outcome)
        {
            // A 类：选棋子升级
            case 1: return 0;      // 赛車   选車   154
            case 2: return 154;    // 炮车   选炮   154
            case 3: return 308;    // 迫击炮 选炮   154
            case 4: return 462;    // 狙击手 选兵   154
            case 6: return 616;    // 巨象   选象   154
            case 7: return 770;    // 小飞象 选象   154
            case 8: return 924;    // 自爆兵 选兵   154
            case 11: return 1078;  // 连环马 选两个马 308
            case 12: return 1386;  // 武士   选士   154

            // A 类：叛变 / 冻结 / 解冻
            case 16: return 1540;                    // 叛变 选敌方棋子 154
            case 28: case 29: case 30: return 1694;  // 冻结 选敌方棋子 154
            case 31: case 32: case 33:
            case 34: case 35: return 1848;           // 解冻 选己方冻结 154

            // B 类：生成
            case 13: return 2002;  // 炮兵   选两个位置 308
            case 14: return 2310;  // 万马奔腾 选两个位置 308
            case 15: return 2618;  // 停車场 选一个位置 154

            // B 类：复活
            case 17: case 18: case 19:
            case 20: case 21: return 2772;  // 起死回生 48

            // C 类（自动）和无效果：无槽位
            default: return -1;
        }
    }

    /// <summary>某个 outcome 的槽位数量</summary>
    public static int GetLotterySlotSize(int outcome)
    {
        switch (outcome)
        {
            case 11: case 13: case 14: return 308;  // 选两个
            case 17: case 18: case 19:
            case 20: case 21: return 48;            // 复活
            case 1: case 2: case 3: case 4: case 6: case 7: case 8:
            case 12: case 16: case 15:
            case 28: case 29: case 30:
            case 31: case 32: case 33: case 34: case 35:
                return 154;                          // 选一个格子
            default: return 0;                       // 无槽位
        }
    }

    /// <summary>抽奖后续选择的绝对索引 = 24332 + 槽位起始 + 槽位内偏移</summary>
    public static int EncodeLotterySlot(int outcome, int slotOffset)
    {
        int start = GetLotterySlotStart(outcome);
        if (start < 0) return -1;
        return StateEncoder.MoveActionSize + StateEncoder.SniperActionSize + start + slotOffset;
    }

    // ================================================================
    //  复活 48 组合编码
    //
    //  顺序：按类型（車马象士炮兵）× 等级（0..max）× 该类型的初始位置。
    //  不含将（将死终局，不复活）。
    // ================================================================
    private static readonly (PieceType type, int level)[] ReviveTypeLevels =
    {
        (PieceType.Rook, 0), (PieceType.Rook, 1),
        (PieceType.Knight, 0), (PieceType.Knight, 1),
        (PieceType.Bishop, 0), (PieceType.Bishop, 1), (PieceType.Bishop, 2), (PieceType.Bishop, 3),
        (PieceType.Guard, 0), (PieceType.Guard, 1),
        (PieceType.Cannon, 0), (PieceType.Cannon, 1), (PieceType.Cannon, 2), (PieceType.Cannon, 3),
        (PieceType.Pawn, 0), (PieceType.Pawn, 1), (PieceType.Pawn, 2), (PieceType.Pawn, 3),
    };  // 共 18 种

    /// <summary>复活槽位总数 = 48</summary>
    public const int ReviveSlotSize = 48;

    /// <summary>把 (类型+等级) 转成复活槽位里的起始偏移</summary>
    public static int GetReviveTypeLevelOffset(PieceType type, int level)
    {
        int offset = 0;
        for (int i = 0; i < ReviveTypeLevels.Length; i++)
        {
            if (ReviveTypeLevels[i].type == type && ReviveTypeLevels[i].level == level)
                return offset;
            // 累加该组合的初始位置数量
            offset += GetInitialPositionCount(ReviveTypeLevels[i].type);
        }
        return -1; // 不存在的组合
    }

    /// <summary>某类型的初始位置数量（红/黑各自相同）</summary>
    public static int GetInitialPositionCount(PieceType type)
    {
        switch (type)
        {
            case PieceType.Rook: return 2;
            case PieceType.Knight: return 2;
            case PieceType.Bishop: return 2;
            case PieceType.Guard: return 2;
            case PieceType.Cannon: return 2;
            case PieceType.Pawn: return 5;
            default: return 0; // 将/墙不复活
        }
    }

    /// <summary>复活槽位内的完整编码：类型+等级+初始位置序号 → 槽位偏移（0~47）</summary>
    public static int EncodeRevive(PieceType type, int level, int positionIndex)
    {
        int offset = GetReviveTypeLevelOffset(type, level);
        if (offset < 0) return -1;
        return offset + positionIndex;
    }

    /// <summary>复活槽位解码：槽位偏移（0~47）→ 类型+等级+初始位置序号</summary>
    public static (PieceType type, int level, int positionIndex) DecodeRevive(int slotOffset)
    {
        int remaining = slotOffset;
        foreach (var (type, level) in ReviveTypeLevels)
        {
            int count = GetInitialPositionCount(type);
            if (remaining < count)
                return (type, level, remaining);
            remaining -= count;
        }
        return (PieceType.Empty, 0, -1); // 越界
    }

    /// <summary>复活绝对索引：24332 + 2772(起死回生起始) + 槽位偏移</summary>
    public static int EncodeReviveAbsolute(PieceType type, int level, int positionIndex)
    {
        int offset = EncodeRevive(type, level, positionIndex);
        if (offset < 0) return -1;
        return EncodeLotterySlot(17, offset); // 起死回生用 outcome 17 代表
    }
}
