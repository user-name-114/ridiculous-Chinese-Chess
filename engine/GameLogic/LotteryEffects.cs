using System.Collections.Generic;
//仅包含部分与全局有关的抽奖内容，其余参考LotteryResolver，或者gamemanager。
//纯数据文件，不含unity相关内容
public static class LotteryEffects
{
    public static void Reverse(Gamestate state)
    {
        float midY = (state.lowerBound + state.upperBound) / 2f;

        // 1. 修改棋盘上所有非空棋子
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty) continue;

                // 阵营翻转
                p.thisTeam = -p.thisTeam;
                // 更新棋子自身坐标（用于后续生成移动规则）
                p.thisy = (int)(2 * midY - p.thisy);
            }
        }

        // 2. 重建棋盘数组——根据棋子自身的新坐标重新填充
        // 先暂存所有棋子及其新坐标
        var piecesToMove = new List<Piece>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty) continue;
                piecesToMove.Add(p);
            }
        }

        // 清空棋盘
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
                state[x, y] = Empty.Instance;

        // 按新坐标放回
        foreach (var piece in piecesToMove)
        {
            state[piece.thisx, piece.thisy] = piece;
        }

        // 3. 处理坟墓
        // 交换红黑坟墓列表
        var temp = state.redGraveyard;
        state.redGraveyard = state.blackGraveyard;
        state.blackGraveyard = temp;

        // 更新坟墓中每个棋子的阵营和坐标（对称到河对岸）
        foreach (var p in state.redGraveyard)
        {
            p.thisTeam = -p.thisTeam;
            p.thisy = (int)(2 * midY - p.thisy);
        }
        foreach (var p in state.blackGraveyard)
        {
            p.thisTeam = -p.thisTeam;
            p.thisy = (int)(2 * midY - p.thisy);
        }

        // currentTeam 保持不变，由 GameManager 在效果结束后切换
    }
    /// 洪水：击杀楚河汉界两边行（y=4 和 y=5）上的所有棋子（墙除外）。
    public static void Flood(Gamestate state)
    {
        int[] floodRows = { 4, 5 }; // 楚河汉界两侧
        foreach (int y in floodRows)
        {
            for (int x = state.leftBound; x <= state.rightBound; x++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty || p.type == PieceType.Wall) continue;
                p.isDead = true;
                state[x, y] = Empty.Instance;
                state.AddToGraveyard(p);
            }
        }
    }
    /// <summary>
    /// 冲锋号-C：双方所有【兵】强制向前一步，擦肩而过不冲突，
    /// 若两兵进入同一格且一过河一未过河则过河者死，未过河者存活。
    /// </summary>
    public static void ChargeBugle(Gamestate state)
    {
        int redForward = 1, blackForward = -1;
        // 1. 收集所有兵
        var allPawns = new List<Pawn>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
                if (state[x, y] is Pawn pawn)
                    allPawns.Add(pawn);
        }

        // 2. 第一阶段：判断每个兵能否移动（忽略目标格上的其他兵）
        var canMove = new Dictionary<Pawn, bool>();
        var targetPos = new Dictionary<Pawn, (int x, int y)>();
        foreach (Pawn pawn in allPawns)
        {
            int forward = (pawn.thisTeam == 1) ? redForward : blackForward;
            int tx = pawn.thisx, ty = pawn.thisy + forward;
            bool blocked = false;

            if (!state.IsValidPosition(tx, ty)) { blocked = true; }
            else
            {
                Piece target = state[tx, ty];
                if (target.type == PieceType.Wall) { blocked = true; }
                else if (target.type != PieceType.Empty && !(target is Pawn))
                {
                    // 静态非兵棋子：己方且不允许友伤时阻挡，否则可以击杀（敌方/友伤开启）
                    if (target.thisTeam == pawn.thisTeam && Piece.friendlyFire != 1)
                        blocked = true;
                }
                // 目标格是兵：暂时忽略，第一轮认为可移动
            }
            canMove[pawn] = !blocked && pawn.frozenTurns == 0;
            targetPos[pawn] = (tx, ty);
        }

        // 3. 第二阶段：处理目标格上有其他兵的情况（擦肩而过 / 踩死不能动的兵）
        var finalMoves = new List<(Pawn pawn, int tx, int ty)>();
        var killedByMove = new List<Piece>();

        foreach (Pawn pawn in allPawns)
        {
            if (!canMove[pawn]) continue;          // 被静态阻挡，留在原地
            (int tx, int ty) = targetPos[pawn];
            Piece target = state[tx, ty];
            bool canGo = true;

            if (target is Pawn targetPawn)
            {
                // 目标格是另一个兵
                if (canMove[targetPawn])            // 那个兵也能移动 → 擦肩而过，本格视为空
                {
                    // 无需操作
                }
                else
                {
                    // 那个兵不能移动，当前兵受阻
                    if (targetPawn.thisTeam == pawn.thisTeam)
                    {
                        if (Piece.friendlyFire != 1)
                            canGo = false;          // 己方且不允许友伤 → 不能移动
                        else
                            killedByMove.Add(targetPawn); // 友伤开启：踩死不能动的己方兵
                    }
                    else
                    {
                        killedByMove.Add(targetPawn);     // 敌方兵不能动：直接踩死
                    }
                }
            }
            // 注：原代码中 else if (target.type != PieceType.Empty) 分支永不执行，已删除

            if (canGo)
                finalMoves.Add((pawn, tx, ty));
        }

        // 4. 处理冲突：两个兵进入同一格（仅可能双方兵目标相同，且不是擦肩而过）
        var targetCount = new Dictionary<(int, int), List<Pawn>>();
        foreach (var move in finalMoves)
        {
            var key = (move.tx, move.ty);
            if (!targetCount.ContainsKey(key))
                targetCount[key] = new List<Pawn>();
            targetCount[key].Add(move.pawn);
        }

        var survivors = new HashSet<Pawn>();
        foreach (var kv in targetCount)
        {
            var list = kv.Value;
            if (list.Count == 1)
            {
                survivors.Add(list[0]);
            }
            else // 冲突，两个兵，一红一黑
            {
                Pawn red = list[0].thisTeam == 1 ? list[0] : list[1];
                Pawn black = list[0].thisTeam == -1 ? list[0] : list[1];

                // 冻结的兵总是被吃，无论在谁的半场
                Pawn deadPawn, livePawn;
                if (red.frozenTurns > 0 && black.frozenTurns == 0)
                {
                    deadPawn = red; livePawn = black;
                }
                else if (black.frozenTurns > 0 && red.frozenTurns == 0)
                {
                    deadPawn = black; livePawn = red;
                }
                else
                {
                    bool redCrossed = targetPos[red].y > 4;
                    deadPawn = redCrossed ? red : black;
                    livePawn = redCrossed ? black : red;
                }

                // 立刻清空死亡兵的旧格子，避免残留引用
                int oldX = deadPawn.thisx, oldY = deadPawn.thisy;
                state[oldX, oldY] = Empty.Instance;

                // 将死亡兵坐标更新到冲突格（以便后续自爆位置正确）
                deadPawn.thisx = targetPos[deadPawn].x;
                deadPawn.thisy = targetPos[deadPawn].y;
                killedByMove.Add(deadPawn);
                survivors.Add(livePawn);
            }
        }

        // 5. 执行移动和击杀
        // 处理所有被标记死亡的棋子（被踩死的兵、冲突死亡的兵）
        foreach (Piece p in killedByMove)
        {
            p.isDead = true;
            state[p.thisx, p.thisy] = Empty.Instance;
            state.AddToGraveyard(p);
        }

        // 清空所有幸存兵的旧位置，再统一放入新位置（避免擦肩而过误杀）
        foreach (Pawn pawn in survivors)
        {
            state[pawn.thisx, pawn.thisy] = Empty.Instance;
        }
        foreach (Pawn pawn in survivors)
        {
            var move = finalMoves.Find(m => m.pawn == pawn);
            if (move.Equals(default)) continue;
            (int tx, int ty) = (move.tx, move.ty);

            // 目标格若仍有非兵棋子（未被提前击杀的静态棋子），补刀
            Piece target = state[tx, ty];
            if (target.type != PieceType.Empty && target.type != PieceType.Wall && !(target is Pawn) && !survivors.Contains(target as Pawn))
            {
                target.isDead = true;
                state[tx, ty] = Empty.Instance;
                state.AddToGraveyard(target);
            }

            pawn.thisx = tx;
            pawn.thisy = ty;
            state[tx, ty] = pawn;
        }
        // 留在原地的兵保持不变
    }
    /// <summary>
    /// 激光炮-C：随机选择一行（0~9，固定范围不随棋盘扩大而改变），
    /// 击杀该行上除【将】和【墙】以外的所有棋子。
    /// </summary>
    /// <summary>激光炮-C：返回随机选择的目标行号</summary>
    public static int LaserCannon(Gamestate state)
    {
        System.Random rng = new System.Random();
        int y = rng.Next(0, 10);
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            Piece p = state[x, y];
            if (p.type == PieceType.Empty || p.type == PieceType.Wall || p.type == PieceType.King)
                continue;
            p.isDead = true;
            state[x, y] = Empty.Instance;
            state.AddToGraveyard(p);
        }
        return y;
    }

    /// <summary>领域展开：棋盘向外扩展一圈（列±1，行±2）</summary>
    public static void ExpandBoard(Gamestate state)
    {
        if (state.isBoardExpanded) return;

        // 保存旧棋盘和旧边界
        int oldLeft = state.leftBound, oldRight = state.rightBound;
        int oldLower = state.lowerBound, oldUpper = state.upperBound;
        Piece[,] oldBoard = new Piece[oldRight - oldLeft + 1, oldUpper - oldLower + 1];
        for (int x = oldLeft; x <= oldRight; x++)
            for (int y = oldLower; y <= oldUpper; y++)
                oldBoard[x - oldLeft, y - oldLower] = state[x, y];

        // 更新边界
        state.leftBound -= 1;
        state.rightBound += 1;
        state.lowerBound -= 2;
        state.upperBound += 2;

        // 创建新数组并初始化为空
        var field = typeof(Gamestate).GetField("board",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Piece[,] newBoard = new Piece[state.rightBound - state.leftBound + 1, state.upperBound - state.lowerBound + 1];
        for (int i = 0; i < newBoard.GetLength(0); i++)
            for (int j = 0; j < newBoard.GetLength(1); j++)
                newBoard[i, j] = Empty.Instance;
        field.SetValue(state, newBoard);

        // 复制旧棋子到新棋盘（逻辑坐标不变）
        for (int x = oldLeft; x <= oldRight; x++)
            for (int y = oldLower; y <= oldUpper; y++)
                state[x, y] = oldBoard[x - oldLeft, y - oldLower];

        state.isBoardExpanded = true;
    }

    /// <summary>领域收缩：恢复原始边界，击杀扩展区域的所有棋子</summary>
    public static List<Piece> ShrinkBoard(Gamestate state)
    {
        var killed = new List<Piece>();
        if (!state.isBoardExpanded) return killed;

        // 击杀所有在原始边界外的棋子
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                if (x < 0 || x > 8 || y < 0 || y > 9)
                {
                    Piece p = state[x, y];
                    if (p.type != PieceType.Empty)
                    {
                        p.isDead = true;
                        state[x, y] = Empty.Instance;
                        state.AddToGraveyard(p);
                        killed.Add(p);
                    }
                }
            }
        }

        // 保存存活棋子
        Piece[,] oldBoard = new Piece[9, 10];
        for (int x = 0; x <= 8; x++)
            for (int y = 0; y <= 9; y++)
                oldBoard[x, y] = state[x, y];

        // 恢复原始边界
        state.leftBound = 0;
        state.rightBound = 8;
        state.lowerBound = 0;
        state.upperBound = 9;

        // 创建新数组
        var field = typeof(Gamestate).GetField("board",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Piece[,] newBoard = new Piece[9, 10];
        for (int x = 0; x <= 8; x++)
            for (int y = 0; y <= 9; y++)
                newBoard[x - state.leftBound, y - state.lowerBound] = oldBoard[x, y];
        field.SetValue(state, newBoard);

        state.isBoardExpanded = false;
        return killed;
    }
}
