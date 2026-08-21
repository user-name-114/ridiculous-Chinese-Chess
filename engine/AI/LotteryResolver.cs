using System;
using System.Collections.Generic;

// ====================================================================
// 抽奖效果解析器。
// 在 MCTS 模拟中随机采样抽奖结果并执行，不依赖 Unity。
//作用：在 AI 模拟对局中，当执行抽奖行动时，自动、无交互地完成所有抽奖效果的目标选择和效果应用。
//它替代了人类玩家的点击选择过程，使得 AI 能在没有 UI 的环境下独立运行完整的抽奖流程。
//升级与生成均随机：使用传入的 rng 随机挑选目标，保证模拟的可复现性。
//遵守友伤规则：生成棋子时检查目标格是否已有己方棋子，若友伤关闭则排除该位置。
//多数量生成（如 2 个炮）：每次都从剩余合法位置中随机取，确保不重复覆盖。
//复活重置状态：复活后清除冻结计时，并重置狙击冷却（设冷却为 1，不可用）。
//自动升级将：御驾亲征直接找唯一帅升级，无需选择。
//边界处理：如果无合法目标，静默返回，不会报错。
// ====================================================================
public static class LotteryResolver
{
    /// <summary>主入口：根据抽奖编号在 state 上执行效果（会修改 state）</summary>
    public static void Resolve(Gamestate state, int outcome, System.Random rng)
    {
        int team = state.currentTeam;

        switch (outcome)
        {
            // ---- Type A: 升级 ----
            case 1: ResolveUpgrade(state, PieceType.Rook, 1, team, rng); break;
            case 2: ResolveUpgrade(state, PieceType.Cannon, 1, team, rng); break;
            case 3: ResolveUpgrade(state, PieceType.Cannon, 2, team, rng); break;
            case 4: ResolveUpgrade(state, PieceType.Pawn, 1, team, rng); break;
            case 5: ResolveAutoUpgradeKing(state, team); break;
            case 6: ResolveUpgrade(state, PieceType.Bishop, 1, team, rng); break;
            case 7: ResolveUpgrade(state, PieceType.Bishop, 2, team, rng); break;
            case 8: ResolveUpgrade(state, PieceType.Pawn, 2, team, rng); break;
            case 11: ResolveUpgradeMultiple(state, PieceType.Knight, 1, team, 2, rng); break;
            case 12: ResolveUpgrade(state, PieceType.Guard, 1, team, rng); break;

            // ---- Type A: 叛变 ----
            case 16: ResolveDefect(state, team, rng); break;

            // ---- Type A: 冻结/解冻 ----
            case 28:
            case 29:
            case 30:
                ResolveFreeze(state, team, rng); break;
            case 31:
            case 32:
            case 33:
            case 34:
            case 35:
                ResolveDefrost(state, team, rng); break;

            // ---- Type B: 生成 ----
            case 13: ResolveGenerateCannons(state, team, rng); break;
            case 14: ResolveGenerateKnights(state, team, rng); break;
            case 15: ResolveGenerateRook(state, team, rng); break;

            // ---- Type B: 复活 ----
            case 17:
            case 18:
            case 19:
            case 20:
            case 21:
                ResolveRevive(state, team, rng); break;

            // ---- Type C: 自动生效 ----
            case 9: PieceGenerator.GeneratePawnsOnRiver(state, team); break;
            case 10: PieceGenerator.GenerateWall(state, team); break;
            case 22: LotteryEffects.Reverse(state); break;
            case 23: LotteryEffects.Flood(state); break;
            case 24: LotteryEffects.ChargeBugle(state); break;
            case 25: LotteryEffects.LaserCannon(state); break;
            case 26: if (!state.isBoardExpanded) LotteryEffects.ExpandBoard(state); break;
            case 27: if (state.isBoardExpanded) LotteryEffects.ShrinkBoard(state); break;

            // ---- 36-40: 未中奖 ----
            default: break;
        }
    }

    // ================================================================
    //  Helper: upgrade level limits
    // ================================================================
    private static int MaxUpgradeLevel(PieceType type)
    {
        switch (type)
        {
            case PieceType.Rook: return 1;
            case PieceType.Knight: return 1;
            case PieceType.Cannon: return 3;
            case PieceType.Bishop: return 3;
            case PieceType.Guard: return 1;
            case PieceType.King: return 2;
            case PieceType.Pawn: return 3;
            default: return 0;
        }
    }

    // ================================================================
    //  Type A helpers
    // ================================================================
    private static void ResolveUpgrade(Gamestate state, PieceType type,
        int effectLevel, int team, System.Random rng)
    {
        var candidates = new List<Piece>();
        int max = MaxUpgradeLevel(type);

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == type && p.thisTeam == team
                    && p.upgradeLevel != effectLevel
                    && p.upgradeLevel + effectLevel <= max)
                    candidates.Add(p);
            }

        if (candidates.Count == 0) return;

        Piece chosen = candidates[rng.Next(candidates.Count)];
        chosen.Upgrade(chosen.upgradeLevel + effectLevel);
    }

    private static void ResolveUpgradeMultiple(Gamestate state, PieceType type,
        int effectLevel, int team, int count, System.Random rng)
    {
        var candidates = new List<Piece>();
        int max = MaxUpgradeLevel(type);

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == type && p.thisTeam == team
                    && p.upgradeLevel != effectLevel
                    && p.upgradeLevel + effectLevel <= max)
                    candidates.Add(p);
            }

        int toPick = Math.Min(count, candidates.Count);
        for (int i = 0; i < toPick; i++)
        {
            int idx = rng.Next(candidates.Count);
            Piece p = candidates[idx];
            p.Upgrade(p.upgradeLevel + effectLevel);
            candidates.RemoveAt(idx); // 不重复选同一个棋子
        }
    }

    private static void ResolveAutoUpgradeKing(Gamestate state, int team)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.King && p.thisTeam == team
                    && p.upgradeLevel < MaxUpgradeLevel(PieceType.King))
                {
                    p.Upgrade(p.upgradeLevel + 1);
                    return;
                }
            }
    }

    private static void ResolveDefect(Gamestate state, int team, System.Random rng)
    {
        var candidates = new List<Piece>();
        int enemyTeam = -team;

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty || p.type == PieceType.Wall) continue;
                if (p.type == PieceType.King) continue;
                if (p.thisTeam == enemyTeam)
                    candidates.Add(p);
            }

        if (candidates.Count == 0) return;

        Piece chosen = candidates[rng.Next(candidates.Count)];
        chosen.Defect();
    }

    private static void ResolveFreeze(Gamestate state, int team, System.Random rng)
    {
        var candidates = new List<Piece>();
        int enemyTeam = -team;

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type != PieceType.Empty && p.type != PieceType.Wall
                    && p.thisTeam == enemyTeam)
                    candidates.Add(p);
            }

        if (candidates.Count == 0) return;

        Piece chosen = candidates[rng.Next(candidates.Count)];
        chosen.frozenTurns = 6;
    }

    private static void ResolveDefrost(Gamestate state, int team, System.Random rng)
    {
        var candidates = new List<Piece>();

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.thisTeam == team && p.frozenTurns > 0)
                    candidates.Add(p);
            }

        if (candidates.Count == 0) return;

        Piece chosen = candidates[rng.Next(candidates.Count)];
        chosen.frozenTurns = 0;
    }

    // ================================================================
    //  Type B helpers
    // ================================================================
    private static void ResolveGenerateCannons(Gamestate state, int team, System.Random rng)
    {
        // 己方兵周围的空格/可覆盖格
        var positions = new List<(int x, int y)>();
        int[] dx = { 0, 0, 1, -1 };
        int[] dy = { 1, -1, 0, 0 };

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type != PieceType.Pawn || p.thisTeam != team) continue;

                for (int d = 0; d < 4; d++)
                {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    if (!state.IsValidPosition(nx, ny)) continue;
                    if (state[nx, ny].type == PieceType.Wall) continue;

                    // 去重
                    if (!positions.Contains((nx, ny)))
                        positions.Add((nx, ny));
                }
            }

        // 过滤：可覆盖的格点（敌方可覆盖，己方需友伤）
        var valid = new List<(int x, int y)>();
        foreach (var (px, py) in positions)
        {
            Piece target = state[px, py];
            if (target.type == PieceType.Empty)
                valid.Add((px, py));
            else if (target.thisTeam != team)
                valid.Add((px, py));
            else if (Piece.friendlyFire == 1)
                valid.Add((px, py));
        }

        int toPick = Math.Min(2, valid.Count);
        for (int i = 0; i < toPick; i++)
        {
            int idx = rng.Next(valid.Count);
            var (gx, gy) = valid[idx];
            PieceGenerator.PlacePieceAt(state, gx, gy, PieceType.Cannon);
            valid.RemoveAt(idx);
        }
    }

    private static void ResolveGenerateKnights(Gamestate state, int team, System.Random rng)
    {
        var candidates = new List<(int x, int y)>();

        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                if (!IsOnFriendlyBorder(x, y, team)) continue;

                Piece target = state[x, y];
                if (target.type == PieceType.Wall) continue;
                if (target.type != PieceType.Empty)
                {
                    if (target.thisTeam == team && Piece.friendlyFire != 1) continue;
                }
                candidates.Add((x, y));
            }

        int toPick = Math.Min(2, candidates.Count);
        for (int i = 0; i < toPick; i++)
        {
            int idx = rng.Next(candidates.Count);
            var (gx, gy) = candidates[idx];
            PieceGenerator.PlacePieceAt(state, gx, gy, PieceType.Knight);
            candidates.RemoveAt(idx);
        }
    }

    private static void ResolveGenerateRook(Gamestate state, int team, System.Random rng)
    {
        var positions = state.GetInitialPositions(PieceType.Rook, team);
        var valid = new List<(int x, int y)>();

        foreach (var (px, py) in positions)
        {
            Piece target = state[px, py];
            if (target.type == PieceType.Wall) continue;
            if (target.type != PieceType.Empty)
            {
                if (target.thisTeam == team && Piece.friendlyFire != 1) continue;
            }
            valid.Add((px, py));
        }

        if (valid.Count == 0) return;

        var (gx, gy) = valid[rng.Next(valid.Count)];
        PieceGenerator.PlacePieceAt(state, gx, gy, PieceType.Rook);
    }

    private static void ResolveRevive(Gamestate state, int team, System.Random rng)
    {
        var graveyard = (team == 1) ? state.redGraveyard : state.blackGraveyard;
        if (graveyard.Count == 0) return;

        // 收集所有 (deadPiece, position) 合法组合
        var options = new List<(Piece piece, int x, int y)>();
        foreach (Piece dead in graveyard)
        {
            var positions = state.GetInitialPositions(dead.type, team);
            foreach (var (px, py) in positions)
            {
                Piece target = state[px, py];
                if (target.type == PieceType.Wall) continue;
                if (target.type != PieceType.Empty)
                {
                    if (target.thisTeam == team && Piece.friendlyFire != 1) continue;
                }
                options.Add((dead, px, py));
            }
        }

        if (options.Count == 0) return;

        var chosen = options[rng.Next(options.Count)];
        PieceGenerator.RevivePiece(state, chosen.piece, chosen.x, chosen.y);

        // 重置冻结和狙击冷却
        chosen.piece.frozenTurns = 0;
        if (chosen.piece is Pawn pawn && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3))
        {
            pawn.sniperCooldown = 2;
            pawn.sniperAvailable = false;
        }
    }

    private static bool IsOnFriendlyBorder(int x, int y, int team)
    {
        if (team == 1)
            return (y >= 0 && y <= 4) && (y == 0 || x == 0 || x == 8);
        else
            return (y >= 5 && y <= 9) && (y == 9 || x == 0 || x == 8);
    }
}
