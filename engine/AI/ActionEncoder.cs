using System;
using System.Collections.Generic;

// ====================================================================
// 动作编码器：把 GameAction 映射到策略头输出的 logits 索引（0~24332）。
//
// 索引布局（与 StateEncoder.TotalActionSize = 24333 一致）：
//   移动：0       ~ 23715   (from × to，各 154 格)
//   狙击：23716   ~ 24331   (from × 方向4)
//   抽奖：24332            (抽奖动作本身)
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

}
