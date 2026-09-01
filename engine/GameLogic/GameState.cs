using System.Collections;
using System.Collections.Generic;

//此文件仅用于获取棋盘状态，不包含移动规则！
//纯数据文件，不含unity相关内容

public class Gamestate
{
    public int leftBound = 0;
    public int rightBound = 8;
    public int upperBound = 9;
    public int lowerBound = 0;
    public bool isBoardExpanded = false;
    public bool prepareModeOn = true;
    public int prepareLotteryCount = 0;

    internal Piece[,] board;
    public int currentTeam = 1; // 红先
    public List<Piece> redGraveyard = new List<Piece>();    // 红方阵亡棋子
    public List<Piece> blackGraveyard = new List<Piece>();  // 黑方阵亡棋子
    public List<(int x, int y)> lianHuanMaTargets = new List<(int x, int y)>();// 连环马跳斩目标列表
    public int lianHuanMaTeam = 0; // 目标所属队伍

    // 索引器：外部用逻辑坐标访问，内部自动转换为数组索引
    public Piece this[int x, int y]
    {
        get => board[x - leftBound, y - lowerBound];
        set => board[x - leftBound, y - lowerBound] = value;
    }

    // ====== 原九宫（用于墙生成 + 绕柱）======
    public (int x, int y) GetOriginalPalaceCenter(int team) => team == 1 ? (4, 1) : (4, 8);
    public int GetOriginalPalaceTop(int team) => team == 1 ? 2 : 9;
    public int GetOriginalPalaceBottom(int team) => team == 1 ? 0 : 7;

    // ====== 当前九宫（用于移动合法性）======
    public int GetCurrentPalaceTop(int team) => team == 1 ? 2 : upperBound;
    public int GetCurrentPalaceBottom(int team) => team == 1 ? lowerBound : 7;

    public void AddLianHuanMaTarget(int x, int y, int team)
    {
        if (lianHuanMaTeam == 0)
            lianHuanMaTeam = team;
        else if (lianHuanMaTeam != team)
            return;

        lianHuanMaTargets.Add((x, y));
    }
    public void ClearLianHuanMaTargets(int team)
    {
        if (lianHuanMaTeam == team)
        {
            lianHuanMaTargets.Clear();
            lianHuanMaTeam = 0;
        }
    }

    public Gamestate()
    {
        board = new Piece[rightBound - leftBound + 1, upperBound - lowerBound + 1];
        for (int x = 0; x <= rightBound; x++)
            for (int y = 0; y <= upperBound; y++)
                this[x, y] = Empty.Instance;

        // 红方 (team = 1) 底线 y=0
        this[0, 0] = new Rook(0, 0, 1);
        this[1, 0] = new Knight(1, 0, 1);
        this[2, 0] = new Bishop(2, 0, 1);
        this[3, 0] = new Guard(3, 0, 1);
        this[4, 0] = new King(4, 0, 1);
        this[5, 0] = new Guard(5, 0, 1);
        this[6, 0] = new Bishop(6, 0, 1);
        this[7, 0] = new Knight(7, 0, 1);
        this[8, 0] = new Rook(8, 0, 1);

        this[1, 2] = new Cannon(1, 2, 1);
        this[7, 2] = new Cannon(7, 2, 1);

        this[0, 3] = new Pawn(0, 3, 1);
        this[2, 3] = new Pawn(2, 3, 1);
        this[4, 3] = new Pawn(4, 3, 1);
        this[6, 3] = new Pawn(6, 3, 1);
        this[8, 3] = new Pawn(8, 3, 1);

        // 黑方 (team = -1) 底线 y=9
        this[0, 9] = new Rook(0, 9, -1);
        this[1, 9] = new Knight(1, 9, -1);
        this[2, 9] = new Bishop(2, 9, -1);
        this[3, 9] = new Guard(3, 9, -1);
        this[4, 9] = new King(4, 9, -1);
        this[5, 9] = new Guard(5, 9, -1);
        this[6, 9] = new Bishop(6, 9, -1);
        this[7, 9] = new Knight(7, 9, -1);
        this[8, 9] = new Rook(8, 9, -1);

        this[1, 7] = new Cannon(1, 7, -1);
        this[7, 7] = new Cannon(7, 7, -1);

        this[0, 6] = new Pawn(0, 6, -1);
        this[2, 6] = new Pawn(2, 6, -1);
        this[4, 6] = new Pawn(4, 6, -1);
        this[6, 6] = new Pawn(6, 6, -1);
        this[8, 6] = new Pawn(8, 6, -1);
    }

    private Gamestate(bool empty)
    {
        board = new Piece[rightBound - leftBound + 1, upperBound - lowerBound + 1];
        for (int x = leftBound; x <= rightBound; x++)
            for (int y = lowerBound; y <= upperBound; y++)
                this[x, y] = Empty.Instance;
    }

    public Gamestate DeepClone()
    {
        Gamestate clone = new Gamestate(true);
        clone.leftBound = this.leftBound;
        clone.rightBound = this.rightBound;
        clone.lowerBound = this.lowerBound;
        clone.upperBound = this.upperBound;
        clone.isBoardExpanded = this.isBoardExpanded;
        clone.currentTeam = this.currentTeam;
        clone.prepareModeOn = this.prepareModeOn;
        clone.prepareLotteryCount = this.prepareLotteryCount;
        clone.lianHuanMaTeam = this.lianHuanMaTeam;   // 修复：连环马跳斩目标归属（DeepClone 此前丢失，搜索克隆体内连环马走法不可见）
        clone.lianHuanMaTargets = new List<(int x, int y)>(this.lianHuanMaTargets);   // 修复(补)：目标列表本体也要复制

        // 用正确的边界重建 board 数组
        Piece[,] newBoard = new Piece[clone.rightBound - clone.leftBound + 1, clone.upperBound - clone.lowerBound + 1];
        for (int i = 0; i < newBoard.GetLength(0); i++)
            for (int j = 0; j < newBoard.GetLength(1); j++)
                newBoard[i, j] = Empty.Instance;
        clone.board = newBoard;

        for (int x = leftBound; x <= rightBound; x++)
            for (int y = lowerBound; y <= upperBound; y++)
                clone[x, y] = this[x, y].Clone();

        foreach (Piece p in redGraveyard)
            clone.redGraveyard.Add(p.Clone());
        foreach (Piece p in blackGraveyard)
            clone.blackGraveyard.Add(p.Clone());

        return clone;
    }

    public bool IsValidPosition(int x, int y)
    {
        return x >= leftBound && x <= rightBound && y >= lowerBound && y <= upperBound;
    }

    public List<(int x, int y)> GetInitialPositions(PieceType type, int team)
    {
        var positions = new List<(int x, int y)>();

        if (team == 1)
        {
            switch (type)
            {
                case PieceType.Rook: positions.Add((0, 0)); positions.Add((8, 0)); break;
                case PieceType.Knight: positions.Add((1, 0)); positions.Add((7, 0)); break;
                case PieceType.Bishop: positions.Add((2, 0)); positions.Add((6, 0)); break;
                case PieceType.Guard: positions.Add((3, 0)); positions.Add((5, 0)); break;
                case PieceType.King: positions.Add((4, 0)); break;
                case PieceType.Cannon: positions.Add((1, 2)); positions.Add((7, 2)); break;
                case PieceType.Pawn:
                    positions.Add((0, 3)); positions.Add((2, 3));
                    positions.Add((4, 3)); positions.Add((6, 3));
                    positions.Add((8, 3)); break;
            }
        }
        else if (team == -1)
        {
            switch (type)
            {
                case PieceType.Rook: positions.Add((0, 9)); positions.Add((8, 9)); break;
                case PieceType.Knight: positions.Add((1, 9)); positions.Add((7, 9)); break;
                case PieceType.Bishop: positions.Add((2, 9)); positions.Add((6, 9)); break;
                case PieceType.Guard: positions.Add((3, 9)); positions.Add((5, 9)); break;
                case PieceType.King: positions.Add((4, 9)); break;
                case PieceType.Cannon: positions.Add((1, 7)); positions.Add((7, 7)); break;
                case PieceType.Pawn:
                    positions.Add((0, 6)); positions.Add((2, 6));
                    positions.Add((4, 6)); positions.Add((6, 6));
                    positions.Add((8, 6)); break;
            }
        }
        return positions;
    }

    private bool IsWuShiGuarded(int x, int y, int movingTeam)
    {
        int[] dx = { 0, 0, 1, -1 };
        int[] dy = { 1, -1, 0, 0 };
        for (int i = 0; i < 4; i++)
        {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (IsValidPosition(nx, ny))
            {
                Piece p = this[nx, ny];
                if (p.type == PieceType.Guard && p.upgradeLevel == 1 && p.thisTeam != movingTeam)
                    return true;
            }
        }
        return false;
    }

    public bool IsBlocked(int x, int y, int movingTeam)
    {
        Piece p = this[x, y];
        if (p.type == PieceType.Empty)
        {
            if (IsWuShiGuarded(x, y, movingTeam))
                return true;
            return false;
        }
        if (p.type == PieceType.Wall && p.thisTeam == movingTeam)
            return false;
        return true;
    }

    public void AddToGraveyard(Piece piece)
    {
        if (piece.thisTeam == 1) redGraveyard.Add(piece);
        else if (piece.thisTeam == -1) blackGraveyard.Add(piece);
    }
    public void RemoveFromGraveyard(Piece piece)
    {
        if (piece.thisTeam == 1)
            redGraveyard.Remove(piece);
        else if (piece.thisTeam == -1)
            blackGraveyard.Remove(piece);
    }

    public void UpdateAllSniperCooldowns()
    {
        for (int x = leftBound; x <= rightBound; x++)
        {
            for (int y = lowerBound; y <= upperBound; y++)
            {
                Piece piece = this[x, y];
                if (piece is Pawn pawn && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3))
                {
                    pawn.UpdateSniperCooldown();
                }
            }
        }
    }
    public void UpdateFrozenTurns()
    {
        for (int x = 0; x <= rightBound; x++)
            for (int y = 0; y <= upperBound; y++)
            {
                Piece p = this[x, y];
                if (p.frozenTurns > 0)
                    p.frozenTurns--;
            }
    }
}
