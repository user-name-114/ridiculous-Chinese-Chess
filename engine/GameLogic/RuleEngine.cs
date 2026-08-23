using System.Collections;
using System.Collections.Generic;
using System;

/// 表示棋盘上的一步合法移动，存储起始和目标坐标。
/// 提供静态方法从当前棋盘状态生成某一方的所有合法移动。
/// //纯数据文件，不含unity相关内容
public class LegalMove
{
    public int x1, y1, x2, y2;   // 起始坐标 (x1,y1) 和目标坐标 (x2,y2)

    public LegalMove(int x1, int y1, int x2, int y2)
    {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    /// 获取当前玩家在给定棋盘状态下的所有合法移动。
    /// <param name="state">当前棋盘状态
    /// <param name="currentTeam">当前行动方阵营（1=红，-1=黑）
    /// 返回合法移动列表
    public static List<LegalMove> GetLegalMoves(Gamestate state, int currentTeam)
    {
        List<LegalMove> legalMoveList = new List<LegalMove>();

        // 遍历棋盘上所有格子
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece piece = state[x, y];

                // 跳过空格和非己方棋子
                if (piece.type == PieceType.Empty || piece.thisTeam != currentTeam)
                    continue;

                // 对于己方棋子，遍历棋盘上所有可能的落点，判断是否合法
                for (int tx = state.leftBound; tx <= state.rightBound; tx++)
                {
                    for (int ty = state.lowerBound; ty <= state.upperBound; ty++) //如果后续有优化需要，这里可以增加判断，给每一类棋子限定搜索范围。
                    {
                        if (piece.IsLegalMove(tx, ty, state))
                        {
                            // 合法则记录该移动
                            legalMoveList.Add(new LegalMove(x, y, tx, ty));
                        }
                    }
                }
            }
        }

        return legalMoveList;
    }
}

public enum PieceType
{
    Empty,
    Rook,
    Knight,
    Bishop,
    Guard,
    King,
    Cannon,
    Pawn,
    Wall
}

public abstract class Piece
{
    public int thisTeam;
    public bool isDead = false;
    public int thisx;
    public int thisy;
    public PieceType type;
    public int upgradeLevel = 0;
    public int frozenTurns = 0;
    public int freezeTickCount = 0;  // 冻结后已执行的EndTurnAndUpdate次数（含对手回合），用于视觉恢复
    public static int friendlyFire = 0;

    public Piece(int x1, int y1, int team, PieceType piecetype)
    {
        thisx = x1;
        thisy = y1;
        thisTeam = team;
        type = piecetype;
    }
    public abstract bool IsLegalMove(int x2, int y2, Gamestate state);
    public void Move(int x2, int y2, Gamestate state)
    {
        // 1.吃子逻辑：非空则标记死亡
        if (state[x2, y2].type != PieceType.Empty)
        {
            state[x2, y2].isDead = true;
        }
        // 2.原位置清空
        state[thisx, thisy] = Empty.Instance;
        // 3.更新自身坐标
        thisx = x2;
        thisy = y2;
        // 4.更新棋盘
        state[thisx, thisy] = this;
    }
    public virtual Piece Clone()
    {
        return (Piece)this.MemberwiseClone();
    }
    public virtual void Upgrade(int newLevel)
    {
        upgradeLevel = newLevel;
    }
    /// 叛变：阵营反转，保留所有状态。
    public void Defect()
    {
        thisTeam = -thisTeam;
    }
    public virtual string GetUpgradeName()
    {
        return "基础";
    }
}

public sealed class Empty : Piece
{
    private static readonly Empty instance = new Empty();
    public static Empty Instance => instance;

    private Empty() : base(-1, -1, 0, PieceType.Empty) { }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        return false;
    }
    public override Piece Clone()
    {
        return Instance;  // 始终返回唯一实例
    }
}

// 车
public class Rook : Piece
{
    public Rook(int x1, int y1, int team) : base(x1, y1, team, PieceType.Rook) { }
    private bool IsLineClear(int x1, int y1, int x2, int y2, Gamestate state)
    {
        // 必须同一直线
        if (x1 != x2 && y1 != y2) return false;
        int dx = 0, dy = 0;
        if (x1 == x2) dy = (y2 > y1) ? 1 : -1;
        else dx = (x2 > x1) ? 1 : -1;

        int cx = x1 + dx, cy = y1 + dy;
        while (cx != x2 || cy != y2) // 直到终点前
        {
            if (state.IsBlocked(cx, cy, thisTeam))
                return false;
            cx += dx;
            cy += dy;
        }
        return true;
    }
    /// 检查赛車的直角拐弯路径是否畅通，并且目标格合法（调用前必须保证拐点坐标合法且为直线）
    private bool TrySaiChePath(int fromX, int fromY, int midX, int midY, int toX, int toY, Gamestate state)
    {
        // 拐点必须为空（且不是起点也不是终点，这里拐点不可能为起点或终点，因为起点和终点不同线）
        if (state.IsBlocked(midX, midY, thisTeam))
            return false;

        // 第一段：from -> mid 路径畅通
        if (!IsLineClear(fromX, fromY, midX, midY, state))
            return false;

        // 第二段：mid -> to 路径畅通
        if (!IsLineClear(midX, midY, toX, toY, state))
            return false;

        // 目标格检查（复用标准逻辑）
        Piece target = state[toX, toY];
        if (target.type == PieceType.Wall) return false; // 墙不可吃
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        // ----- 原版规则（升级等级为0） -----
        /// 以下判断从当前位置移动到 (x2, y2) 是否符合车的移动规则
        if (upgradeLevel == 0)
        {
            // 1. 目标位置必须在棋盘范围内
            if (!state.IsValidPosition(x2, y2)) return false;

            // 2. 必须沿直线移动（横坐标相同或纵坐标相同）
            //    如果 x 和 y 都不相等，说明不是直线，非法
            if (thisx != x2 && thisy != y2) return false;
            //如果目标位置和当前位置相同，则不合法（不算移动）
            if (thisx == x2 && thisy == y2) return false;

            // 3. 计算直线上的移动方向
            int dx = 0, dy = 0;
            if (thisx == x2)
                // 纵向移动，dy 为向上或向下一步
                dy = (y2 > thisy) ? 1 : -1;
            else
                // 横向移动，dx 为向左或向右一步
                dx = (x2 > thisx) ? 1 : -1;

            // 4. 检查从起点到目标点之间的路径是否有棋子阻挡（不包括目标格）
            int cx = thisx + dx, cy = thisy + dy;
            while (cx != x2 || cy != y2)   // 只要还没到达目标格
            {
                if (state.IsBlocked(cx, cy, thisTeam))
                    return false;          // 路径上有棋子，阻挡移动
                cx += dx;
                cy += dy;
            }

            // 5. 判断目标格子是否可进入（结合友伤设置）
            Piece target = state[x2, y2];
            if (target.type == PieceType.Wall) return false;       // 墙不可吃
            if (target.type == PieceType.Empty) return true;       // 空格，可以移动
            if (target.thisTeam != thisTeam) return true;          // 敌方棋子，可以吃
            // 目标为友军：仅当全局允许友军伤害（friendlyFire == 1）时合法
            return friendlyFire == 1;
        }
        else if (upgradeLevel == 1) // 赛車
        {
            // 1. 边界检查
            if (!state.IsValidPosition(x2, y2)) return false;

            // 2. 不能原地移动
            if (thisx == x2 && thisy == y2) return false;

            // 3. 直线移动（不拐弯）
            if (thisx == x2 || thisy == y2)
            {
                if (IsLineClear(thisx, thisy, x2, y2, state))
                {
                    // 4. 目标格检查
                    Piece target = state[x2, y2];
                    if (target.type == PieceType.Empty) return true;
                    if (target.thisTeam != thisTeam) return true;
                    return friendlyFire == 1;
                }
                return false;
            }
            else // 4. 拐弯移动
            {
                // 路径1：先纵向后横向，拐点 (thisx, y2)
                if (TrySaiChePath(thisx, thisy, thisx, y2, x2, y2, state))
                    return true;
                // 路径2：先横向后纵向，拐点 (x2, thisy)
                if (TrySaiChePath(thisx, thisy, x2, thisy, x2, y2, state))
                    return true;
                return false;
            }
        }
        // 其他 upgradeLevel 返回 false
        return false;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "车",
            1 => "赛车",
            _ => "未知升级",
        };
    }
}

// 马
public class Knight : Piece
{
    public Knight(int x1, int y1, int team) : base(x1, y1, team, PieceType.Knight) { }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        // 基础马步（lv0与lv1均保留）
        if (upgradeLevel == 0 || upgradeLevel == 1)
        {
            if (!state.IsValidPosition(x2, y2)) return false;

            int dx = x2 - thisx, dy = y2 - thisy;
            bool isKnightMove = (Math.Abs(dx) == 1 && Math.Abs(dy) == 2) ||
                                (Math.Abs(dx) == 2 && Math.Abs(dy) == 1);
            if (isKnightMove)
            {
                int legX = thisx, legY = thisy;
                if (Math.Abs(dx) == 2) { legX = thisx + dx / 2; legY = thisy; }
                else { legX = thisx; legY = thisy + dy / 2; }

                if (state.IsBlocked(legX, legY, thisTeam)) return false;

                Piece target = state[x2, y2];
                if (target.type == PieceType.Wall) return false;
                if (target.type == PieceType.Empty) return true;
                if (target.thisTeam != thisTeam) return true;
                return friendlyFire == 1;
            }
        }

        // 连环马跳斩：目标位置必须存在于列表中
        if (upgradeLevel == 1 &&
            state.lianHuanMaTeam == thisTeam &&
            state.lianHuanMaTargets.Contains((x2, y2)))
        {
            Piece target = state[x2, y2];
            if (target.type == PieceType.Wall) return false;
            if (target.type == PieceType.Empty) return true;
            if (target.thisTeam != thisTeam) return true;
            return friendlyFire == 1;
        }

        return false;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "马",
            1 => "连环马",
            _ => "未知升级",
        };
    }
}

// 象
public class Bishop : Piece
{
    public Bishop(int x1, int y1, int team) : base(x1, y1, team, PieceType.Bishop) { }
    /// 通用斜向走法检查
    /// <param name="step">斜跨格数（2 或 3）</param>
    /// <param name="checkRiver">是否检查过河限制</param>
    private bool IsValidDiagonalMove(int step, int x2, int y2, Gamestate state, bool checkRiver)
    {
        int dx = x2 - thisx, dy = y2 - thisy;
        if (Math.Abs(dx) != step || Math.Abs(dy) != step) return false;

        // 过河限制（仅当 checkRiver 为 true 时生效）
        if (checkRiver)
        {
            if (thisTeam == 1 && y2 > 4) return false;
            if (thisTeam == -1 && y2 < 5) return false;
        }

        // 检查路径上的所有中间点
        int signX = Math.Sign(dx);
        int signY = Math.Sign(dy);
        for (int i = 1; i < step; i++)
        {
            int midX = thisx + i * signX;
            int midY = thisy + i * signY;
            if (state.IsBlocked(midX, midY, thisTeam))
                return false;
        }

        // 目标格检查
        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false;
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        if (!state.IsValidPosition(x2, y2)) return false;

        if (upgradeLevel == 0) // 普通象：田字，不能过河
        {
            return IsValidDiagonalMove(2, x2, y2, state, true);
        }
        else if (upgradeLevel == 1) // 巨象：田字或 3 格斜线，不能过河
        {
            if (IsValidDiagonalMove(2, x2, y2, state, true)) return true;
            if (IsValidDiagonalMove(3, x2, y2, state, true)) return true;
            return false;
        }
        else if (upgradeLevel == 2) // 小飞象：田字，可过河
        {
            return IsValidDiagonalMove(2, x2, y2, state, false);
        }
        else if (upgradeLevel == 3) // 巨飞象：田字或 3 格斜线，可过河
        {
            if (IsValidDiagonalMove(2, x2, y2, state, false)) return true;
            if (IsValidDiagonalMove(3, x2, y2, state, false)) return true;
            return false;
        }
        return false;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "象",
            1 => "巨象",
            2 => "小飞象",
            3 => "巨飞象",
            _ => "未知升级",
        };
    }
}

// 士
public class Guard : Piece
{
    public Guard(int x1, int y1, int team) : base(x1, y1, team, PieceType.Guard) { }
    /// 判断士的移动是否合法
    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        // ----- 原版规则 -----
        // 1. 目标位置必须在棋盘范围内
        if (!state.IsValidPosition(x2, y2)) return false;

        // 2. 必须斜走1步或2步（2步为穿墙跳到斜对角）
        int dx = Math.Abs(x2 - thisx), dy = Math.Abs(y2 - thisy);
        if (dx != dy || (dx != 1 && dx != 2)) return false;

        // 3. 必须在九宫格内：横坐标限定在 3~5 列
        if (x2 < 3 || x2 > 5) return false;

        // 4. 纵坐标限制：使用当前九宫范围（领域展开后可进入扩展区域）
        if (y2 < state.GetCurrentPalaceBottom(thisTeam) || y2 > state.GetCurrentPalaceTop(thisTeam)) return false;

        // 5. 2步斜走仅当中间格为己方墙时才允许
        if (dx == 2)
        {
            int midX = (thisx + x2) / 2;
            int midY = (thisy + y2) / 2;
            Piece midPiece = state[midX, midY];
            if (midPiece.type != PieceType.Wall || midPiece.thisTeam != thisTeam)
                return false;
        }

        // 6. 目标格判断：墙不可进入，空格或敌方棋子可进入，己方棋子需开启友伤
        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false;
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "士",
            1 => "武士",
            _ => "未知升级",
        };
    }
}

// 将/帅
public class King : Piece
{
    public King(int x1, int y1, int team) : base(x1, y1, team, PieceType.King) { }

    // ----- 原有辅助方法（IsPathClear、CanFlyGeneral、IsTargetValidForRookMove）保持不变 -----
    private bool IsPathClear(int x2, int y2, Gamestate state)
    {
        if (thisx != x2 && thisy != y2) return false;
        int dx = 0, dy = 0;
        if (thisx == x2) dy = (y2 > thisy) ? 1 : -1;
        else dx = (x2 > thisx) ? 1 : -1;
        int cx = thisx + dx, cy = thisy + dy;
        while (cx != x2 || cy != y2)
        {
            if (state.IsBlocked(cx, cy, thisTeam))
                return false;
            cx += dx;
            cy += dy;
        }
        return true;
    }

    private bool CanFlyGeneral(int x2, int y2, Gamestate state)
    {
        if (thisx != x2) return false;
        Piece target = state[x2, y2];
        if (target.type != PieceType.King || target.thisTeam == thisTeam)
            return false;
        int step = (y2 > thisy) ? 1 : -1;
        int cy = thisy + step;
        while (cy != y2)
        {
            if (state.IsBlocked(thisx, cy, thisTeam))
                return false;
            cy += step;
        }
        return true;
    }

    private bool IsTargetValidForRookMove(int x2, int y2, Gamestate state)
    {
        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false;
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }

    // ----- 新增：绕柱移动判断 -----
    /// <summary>
    /// 绕己方九宫中心墙顺/逆时针移动任意步（不可穿过其他棋子）
    /// </summary>
    private bool IsCircleMove(int x2, int y2, Gamestate state)
    {
        // 确定中心墙位置
        int centerX, centerY;
        if (thisTeam == 1) { centerX = 4; centerY = 1; }
        else { centerX = 4; centerY = 8; }

        // 中心必须为己方墙
        Piece centerPiece = state[centerX, centerY];
        if (centerPiece.type != PieceType.Wall || centerPiece.thisTeam != thisTeam)
            return false;

        // 确定该阵营的环序列（顺时针）
        int[,] ring;
        if (thisTeam == 1)
            ring = new int[,] { { 3, 0 }, { 4, 0 }, { 5, 0 }, { 5, 1 }, { 5, 2 }, { 4, 2 }, { 3, 2 }, { 3, 1 } };
        else
            ring = new int[,] { { 3, 7 }, { 4, 7 }, { 5, 7 }, { 5, 8 }, { 5, 9 }, { 4, 9 }, { 3, 9 }, { 3, 8 } };

        // 查找起点和终点在环中的索引
        int startIdx = -1, endIdx = -1;
        for (int i = 0; i < 8; i++)
        {
            if (ring[i, 0] == thisx && ring[i, 1] == thisy) startIdx = i;
            if (ring[i, 0] == x2 && ring[i, 1] == y2) endIdx = i;
        }
        if (startIdx == -1 || endIdx == -1 || startIdx == endIdx) return false;

        // 检查两个方向路径是否通畅（不含起点，含终点前的格子）
        // 顺时针：索引递增，模8
        bool clockwiseOK = true;
        for (int i = 1; i <= (endIdx - startIdx + 8) % 8; i++)
        {
            int idx = (startIdx + i) % 8;
            int sx = ring[idx, 0], sy = ring[idx, 1];
            if (sx == x2 && sy == y2) break; // 到达终点，停止检查路径
            if (state.IsBlocked(sx, sy, thisTeam))
            {
                clockwiseOK = false;
                break;
            }
        }

        // 逆时针：索引递减，模8
        bool counterClockwiseOK = true;
        for (int i = 1; i <= (startIdx - endIdx + 8) % 8; i++)
        {
            int idx = (startIdx - i + 8) % 8;
            int sx = ring[idx, 0], sy = ring[idx, 1];
            if (sx == x2 && sy == y2) break;
            if (state.IsBlocked(sx, sy, thisTeam))
            {
                counterClockwiseOK = false;
                break;
            }
        }

        // 任一方向畅通且目标格合法
        return (clockwiseOK || counterClockwiseOK) && IsTargetValidForRookMove(x2, y2, state);
    }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        if (!state.IsValidPosition(x2, y2)) return false;

        // 通用：直线穿墙两格（已实现）
        int dx = x2 - thisx, dy = y2 - thisy;
        if ((Math.Abs(dx) == 2 && dy == 0) || (dx == 0 && Math.Abs(dy) == 2))
        {
            int midX = (thisx + x2) / 2, midY = (thisy + y2) / 2;
            Piece midPiece = state[midX, midY];
            if (midPiece.type == PieceType.Wall && midPiece.thisTeam == thisTeam)
            {
                if (x2 >= 3 && x2 <= 5 &&
                    (y2 >= state.GetCurrentPalaceBottom(thisTeam) && y2 <= state.GetCurrentPalaceTop(thisTeam)))
                    return IsTargetValidForRookMove(x2, y2, state);
            }
        }

        if (upgradeLevel == 0)
        {
            dx = Math.Abs(x2 - thisx); dy = Math.Abs(y2 - thisy);
            if ((dx == 1 && dy == 0) || (dx == 0 && dy == 1))
            {
                if (x2 >= 3 && x2 <= 5 &&
                    (y2 >= state.GetCurrentPalaceBottom(thisTeam) && y2 <= state.GetCurrentPalaceTop(thisTeam)))
                    return IsTargetValidForRookMove(x2, y2, state);
            }
            if (CanFlyGeneral(x2, y2, state)) return true;
        }
        else if (upgradeLevel >= 1) // 将军(lv1) / 飞将(lv2)
        {
            if (thisx == x2 && thisy == y2) return false;
            if ((thisx == x2 || thisy == y2) && IsPathClear(x2, y2, state))
            {
                int dist = Math.Abs(thisx - x2) + Math.Abs(thisy - y2);
                int maxRange = upgradeLevel == 1 ? 2 : 3;
                if (dist <= maxRange)
                {
                    bool crossRiver = (thisTeam == 1 && y2 > 4) || (thisTeam == -1 && y2 < 5);
                    if (!crossRiver)
                        return IsTargetValidForRookMove(x2, y2, state);
                }
            }
            if (CanFlyGeneral(x2, y2, state)) return true;
        }

        // 绕柱移动（所有升级通用）
        if (IsCircleMove(x2, y2, state))
            return true;

        return false;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "将",
            1 => "将军",
            2 => "飞将",
            _ => "未知升级",
        };
    }
}

// 炮
public class Cannon : Piece
{
    // 构造函数
    public Cannon(int x1, int y1, int team) : base(x1, y1, team, PieceType.Cannon) { }
    /// <summary>车移动规则：直线无障碍，空格可进，可吃敌/友（根据友伤）</summary>
    private bool IsValidRookMove(int x2, int y2, Gamestate state)
    {
        // 1. 边界与直线判定
        if (!state.IsValidPosition(x2, y2)) return false;
        if (thisx != x2 && thisy != y2) return false;
        if (thisx == x2 && thisy == y2) return false;

        // 2. 方向与路径阻挡检查
        int dx = 0, dy = 0;
        if (thisx == x2)
            dy = (y2 > thisy) ? 1 : -1;
        else
            dx = (x2 > thisx) ? 1 : -1;

        int cx = thisx + dx, cy = thisy + dy;
        while (cx != x2 || cy != y2)
        {
            if (state.IsBlocked(cx, cy, thisTeam)) // 己方墙可穿过，敌方墙阻挡
                return false;
            cx += dx;
            cy += dy;
        }

        // 3. 目标格检查
        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false; // 墙不可吃
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }

    /// 炮规则：移动时路径空；吃子时路径棋子数在 1～intervals 之间。
    /// <param name="intervals">吃子时允许的最大炮架数量（含）</param>
    private bool IsValidCannonMove(int x2, int y2, Gamestate state, int intervals)
    {
        if (!state.IsValidPosition(x2, y2)) return false;
        if (thisx != x2 && thisy != y2) return false;
        if (thisx == x2 && thisy == y2) return false;

        int dx = 0, dy = 0;
        if (thisx == x2) dy = (y2 > thisy) ? 1 : -1;
        else dx = (x2 > thisx) ? 1 : -1;

        int count = 0;
        int cx = thisx + dx, cy = thisy + dy;
        while (cx != x2 || cy != y2)
        {
            Piece p = state[cx, cy];
            if (p.type == PieceType.Wall)
            {
                // 墙不可跨越，无论敌我，直接阻挡（不能攻击墙后棋子，也不能移动跨墙）
                return false;
            }
            else if (p.type != PieceType.Empty)
            {
                count++;
            }
            cx += dx;
            cy += dy;
        }

        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false; // 不可吃墙
        if (target.type == PieceType.Empty)
        {
            return count == 0;          // 移动时不能有棋子
        }
        else
        {
            if (count < 1 || count > intervals) return false; // 炮架数必须合法
            if (target.thisTeam != thisTeam) return true;
            return friendlyFire == 1;
        }
    }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        if (upgradeLevel == 0) // 普通炮
        {
            return IsValidCannonMove(x2, y2, state, 1);
        }
        else if (upgradeLevel == 1) // 炮车
        {
            if (IsValidRookMove(x2, y2, state)) return true;
            return IsValidCannonMove(x2, y2, state, 1);
        }
        else if (upgradeLevel == 2) // 迫击炮
        {
            return IsValidCannonMove(x2, y2, state, 2);
        }
        else if (upgradeLevel == 3) // 迫击炮车（炮车+迫击炮叠加）
        {
            if (IsValidRookMove(x2, y2, state)) return true;
            return IsValidCannonMove(x2, y2, state, 2);
        }
        return false;
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "炮",
            1 => "炮车",
            2 => "迫击炮",
            3 => "迫击炮车",
            _ => "未知升级",
        };
    }
}

// 兵
public class Pawn : Piece
{
    // 狙击手状态字段
    public int sniperCooldown = 0;      // 剩余冷却步数（0=冷却完毕）
    public bool sniperAvailable = false; // 当前是否有狙击机会
    public bool canExplode = false;   // 自爆兵标记，升级为自爆兵时设为 true

    public Pawn(int x1, int y1, int team) : base(x1, y1, team, PieceType.Pawn) { }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        if (frozenTurns > 0) return false;
        // 1. 目标必须在棋盘内
        if (!state.IsValidPosition(x2, y2)) return false;

        int dx = x2 - thisx, dy = y2 - thisy;

        // 2. 分阵营判断移动方向
        if (thisTeam == 1) // 红方（向前为 y 增加）
        {
            // 不能后退（dy 必须非负，dy=0 为过河后平移）
            if (dy < 0) return false;

            if (thisy <= 4) // 未过河：只能在河界（y=4）及之前直走一步
            {
                if (dx == 0 && dy == 1) { /* 合法，继续检查目标 */ }
                else return false;
            }
            else // 过河后：可向前、左、右各走一步
            {
                if (!((dx == 0 && dy == 1) || (Math.Abs(dx) == 1 && dy == 0)))
                    return false;
            }
        }
        else // 黑方（向前为 y 减少）
        {
            // 不能后退（dy 必须非正，dy=0 为过河后平移）
            if (dy > 0) return false;

            if (thisy >= 5) // 未过河：只能在河界（y=5）及之前直走一步
            {
                if (dx == 0 && dy == -1) { /* 合法，继续检查目标 */ }
                else return false;
            }
            else // 过河后：可向前、左、右各走一步
            {
                if (!((dx == 0 && dy == -1) || (Math.Abs(dx) == 1 && dy == 0)))
                    return false;
            }
        }

        // 3. 目标格检查：空格、敌方棋子均可；友军则需开启友伤
        Piece target = state[x2, y2];
        if (target.type == PieceType.Wall) return false;
        if (target.type == PieceType.Empty) return true;
        if (target.thisTeam != thisTeam) return true;
        return friendlyFire == 1;
    }
    /// <summary>
    /// 执行自爆：击杀周围 8 格内符合条件的棋子（墙不可摧毁，友伤关闭时不炸友军）
    /// 返回所有被炸死的棋子列表，供 GameManager 进行视觉和坟墓处理
    public List<Piece> Explode(Gamestate state)
    {
        List<Piece> killed = new List<Piece>();
        // 8 个相邻方向
        int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };

        for (int i = 0; i < 8; i++)
        {
            int nx = thisx + dx[i];
            int ny = thisy + dy[i];
            if (!state.IsValidPosition(nx, ny)) continue;

            Piece target = state[nx, ny];
            if (target.type == PieceType.Empty) continue; // 墙不可摧毁
            if (friendlyFire == 0 && target.thisTeam == thisTeam) continue; // 友伤关闭时不炸友军

            // 击杀目标
            target.isDead = true;
            state[nx, ny] = Empty.Instance;
            killed.Add(target);
        }
        return killed;
    }
    /// 沿指定方向执行狙击：找到并击杀第一个可狙击目标，清除棋盘格，更新冷却
    public void Snipe(int dx, int dy, Gamestate state)
    {
        Piece target = GetSnipeTarget(dx, dy, state);
        if (target != null)
        {
            target.isDead = true;
            state[target.thisx, target.thisy] = Empty.Instance; // 清空格点
            AfterSnipe();
        }
    }
    ///以下为狙击手功能相关方法
    /// 获取该方向上的合法狙击目标（考虑友伤开关）。返回第一个可被狙击的棋子，若无则返回null。
    public Piece GetSnipeTarget(int dx, int dy, Gamestate state)
    {
        if (Math.Abs(dx) + Math.Abs(dy) != 1) return null;

        int cx = thisx + dx, cy = thisy + dy;
        while (state.IsValidPosition(cx, cy))
        {
            Piece target = state[cx, cy];
            if (target.type == PieceType.Wall) return null; // 墙阻断视线
            if (target.type != PieceType.Empty)
            {
                if (friendlyFire == 1) return target;
                if (target.thisTeam != thisTeam) return target;
                // 友伤关闭时，友方棋子不阻挡视线，继续搜索
            }
            cx += dx;
            cy += dy;
        }
        return null; // 方向上无棋子
    }
    /// 检查该方向是否能狙击（调用 GetSnipeTarget）
    public bool CanSnipeInDirection(int dx, int dy, Gamestate state)
    {
        return GetSnipeTarget(dx, dy, state) != null;
    }
    /// 执行狙击后的状态更新：消耗狙击机会，进入4步冷却
    public void AfterSnipe()
    {
        sniperAvailable = false;
        sniperCooldown = 4;
    }
    /// 回合结束时更新冷却（由GameManager调用）
    public void UpdateSniperCooldown()
    {
        if (sniperCooldown > 0)
        {
            sniperCooldown--;
            if (sniperCooldown == 0)
                sniperAvailable = true; // 冷却完毕，获得狙击机会
        }
    }
    // 在 Pawn 类中，重写基类方法
    public override void Upgrade(int newLevel)
    {
        bool hadSniper = (upgradeLevel == 1 || upgradeLevel == 3);

        base.Upgrade(newLevel); // 设置 upgradeLevel

        // 狙击能力管理
        if (newLevel == 1 || newLevel == 3)
        {
            if (!hadSniper)
            {
                sniperCooldown = 4;
                sniperAvailable = false;
            }
            // 已有狙击则冷却状态不变
        }
        else
        {
            sniperCooldown = 0;
            sniperAvailable = false;
        }

        // 自爆能力
        canExplode = (newLevel == 2 || newLevel == 3);
    }
    public override string GetUpgradeName()
    {
        return upgradeLevel switch
        {
            0 => "兵",
            1 => "狙击兵",
            2 => "自爆兵",
            3 => "狙击自爆兵",
            _ => "未知升级",
        };
    }
}

public sealed class Wall : Piece
{
    public int wallDuration = 10; // 剩余持续步数，每次EndTurnAndUpdate减1，为0时消失

    public Wall(int x, int y, int team) : base(x, y, team, PieceType.Wall) { }

    public override bool IsLegalMove(int x2, int y2, Gamestate state)
    {
        return false; // 墙永远不能移动
    }
}