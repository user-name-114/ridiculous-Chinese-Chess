using System.Collections;
using System.Collections.Generic;
//纯数据文件，不含unity相关内容
public static class PieceGenerator
{
    /// <summary>
    /// 秦王绕柱-C：在己方九宫中心生成墙。
    /// </summary>
    public static void GenerateWall(Gamestate state, int team)
    {
        int wx, wy;
        if (team == 1) { wx = 4; wy = 1; }
        else if (team == -1) { wx = 4; wy = 8; }
        else return;

        Piece target = state[wx, wy];
        if (target.type == PieceType.Wall) return; // 已有墙

        if (target.type != PieceType.Empty)
        {
            if (target.thisTeam != team)
                KillAndClear(state, target);
            else if (Piece.friendlyFire == 1)
                KillAndClear(state, target);
            else
                return;
        }

        state[wx, wy] = new Wall(wx, wy, team);
    }

    /// <summary>
    /// 捅了老窝-C：在己方河边的每个格点各生成一个兵。
    /// </summary>
    public static void GeneratePawnsOnRiver(Gamestate state, int team)
    {
        if (team != 1 && team != -1) return;
        int y = (team == 1) ? 4 : 5; // 红方 y=4，黑方 y=5

        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            Piece target = state[x, y];
            if (target.type == PieceType.Wall) continue; // 墙不能覆盖

            if (target.type != PieceType.Empty)
            {
                if (target.thisTeam != team)
                    KillAndClear(state, target);
                else if (Piece.friendlyFire == 1)
                    KillAndClear(state, target);
                else
                    continue; // 不允许友伤，跳过该格
            }

            // 生成己方兵
            state[x, y] = new Pawn(x, y, team);
        }
    }

    /// 击杀棋子并清空格点，放入对应坟墓。
    private static void KillAndClear(Gamestate state, Piece piece)
    {
        piece.isDead = true;
        state[piece.thisx, piece.thisy] = Empty.Instance;
        state.AddToGraveyard(piece);
    }

    /// 在指定位置生成一个指定类型的新棋子（lv0），击杀原有棋子并加入坟墓。
    public static Piece PlacePieceAt(Gamestate state, int x, int y, PieceType type)
    {
        int team = state.currentTeam;
        Piece target = state[x, y];

        // 若目标格有棋子（非墙），则击杀并清理
        if (target.type != PieceType.Empty && target.type != PieceType.Wall)
        {
            KillAndClear(state, target);
        }
        else if (target.type == PieceType.Wall)
        {
            // 墙直接被覆盖（仅特殊情况，调用方应避免）
            state[x, y] = Empty.Instance;
        }

        // 生成新棋子
        Piece newPiece;
        switch (type)
        {
            case PieceType.Rook: newPiece = new Rook(x, y, team); break;
            case PieceType.Knight: newPiece = new Knight(x, y, team); break;
            case PieceType.Cannon: newPiece = new Cannon(x, y, team); break;
            case PieceType.Pawn: newPiece = new Pawn(x, y, team); break;
            default: return null;
        }
        state[x, y] = newPiece;
        return newPiece;
    }
    /// 复活棋子：重置死亡状态，移出坟墓，放置到目标坐标。
    /// 若目标坐标有棋子（非墙），则击杀并入坟。调用方需确保目标坐标合法（非墙等）。
    public static void RevivePiece(Gamestate state, Piece piece, int x, int y)
    {
        // 1. 目标格为墙则无法复活（安全保护）
        Piece target = state[x, y];
        if (target.type == PieceType.Wall) return;

        // 2. 处理目标格原棋子（若有则击杀）
        if (target.type != PieceType.Empty)
        {
            // KillAndClear 会处理死亡、入坟、清空
            KillAndClear(state, target);
        }
        else
        {
            // 若为空，需手动清空（KillAndClear 不处理空格子）
            state[x, y] = Empty.Instance;
        }

        // 3. 复活棋子数据更新
        piece.isDead = false;
        state.RemoveFromGraveyard(piece);
        piece.thisx = x;
        piece.thisy = y;
        state[x, y] = piece;
    }
    /// <summary>
    /// 判断某个位置是否可以生成指定类型的棋子（供生成模式蓝点使用）。
    /// 不修改任何数据，只做可行性检查。
    /// </summary>
    public static bool IsPositionValidForGeneration(Gamestate state, int x, int y, PieceType type, int team)
    {
        if (!state.IsValidPosition(x, y)) return false;

        Piece target = state[x, y];
        if (target.type == PieceType.Wall) return false;

        // 占用规则：敌方可覆盖，己方需友伤开启
        if (target.type != PieceType.Empty)
        {
            if (target.thisTeam == team && Piece.friendlyFire != 1)
                return false;
        }

        // 特殊位置限制
        switch (type)
        {
            case PieceType.Cannon:
                return HasAdjacentFriendlyPawn(state, x, y, team);
            case PieceType.Knight:
                return IsOnFriendlyBorder(x, y, team);
            case PieceType.Rook:
                return state.GetInitialPositions(PieceType.Rook, team).Contains((x, y));
            default:
                return false;
        }
    }

    private static bool HasAdjacentFriendlyPawn(Gamestate state, int x, int y, int team)
    {
        int[] dx = { 0, 0, 1, -1 }, dy = { 1, -1, 0, 0 };
        for (int i = 0; i < 4; i++)
        {
            int nx = x + dx[i], ny = y + dy[i];
            if (state.IsValidPosition(nx, ny) &&
                state[nx, ny].type == PieceType.Pawn &&
                state[nx, ny].thisTeam == team)
                return true;
        }
        return false;
    }

    private static bool IsOnFriendlyBorder(int x, int y, int team)
    {
        if (team == 1) return (y >= 0 && y <= 4) && ((y == 0) || (x == 0) || (x == 8));
        else return (y >= 5 && y <= 9) && ((y == 9) || (x == 0) || (x == 8));
    }
}
