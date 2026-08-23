using System;
using System.Collections.Generic;

// ====================================================================
// 抽奖效果解析器。
// 在 MCTS 模拟中解析抽奖结果并应用效果，不依赖 Unity。
// 主动目标可由调用方枚举并指定；旧 Resolve 接口保留随机回退路径。
//遵守友伤规则：生成棋子时检查目标格是否已有己方棋子，若友伤关闭则排除该位置。
//多数量生成（如 2 个炮）：每次都从剩余合法位置中随机取，确保不重复覆盖。
//复活重置状态：复活后清除冻结计时，并重置狙击冷却（设冷却为 1，不可用）。
//自动升级将：御驾亲征直接找唯一帅升级，无需选择。
//边界处理：如果无合法目标，静默返回，不会报错。
// ====================================================================
public enum LotteryChoiceType
{
    Piece,
    TwoPieces,
    Position,
    TwoPositions,
    Revive
}

public sealed class LotteryChoice
{
    public LotteryChoiceType type;
    public int x;
    public int y;
    public int secondX;
    public int secondY;
    public int graveyardIndex;

    public LotteryChoice(LotteryChoiceType type, int x = 0, int y = 0,
        int secondX = int.MinValue, int secondY = int.MinValue, int graveyardIndex = -1)
    {
        this.type = type;
        this.x = x;
        this.y = y;
        this.secondX = secondX;
        this.secondY = secondY;
        this.graveyardIndex = graveyardIndex;
    }
}

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

    public static List<LotteryChoice> GetChoices(Gamestate state, int outcome)
    {
        int team = state.currentTeam;
        var choices = new List<LotteryChoice>();

        switch (outcome)
        {
            case 1: AddPieceChoices(state, choices, PieceType.Rook, 1, team); break;
            case 2: AddPieceChoices(state, choices, PieceType.Cannon, 1, team); break;
            case 3: AddPieceChoices(state, choices, PieceType.Cannon, 2, team); break;
            case 4: AddPieceChoices(state, choices, PieceType.Pawn, 1, team); break;
            case 6: AddPieceChoices(state, choices, PieceType.Bishop, 1, team); break;
            case 7: AddPieceChoices(state, choices, PieceType.Bishop, 2, team); break;
            case 8: AddPieceChoices(state, choices, PieceType.Pawn, 2, team); break;
            case 11: AddTwoPieceChoices(state, choices, PieceType.Knight, 1, team); break;
            case 12: AddPieceChoices(state, choices, PieceType.Guard, 1, team); break;
            case 16: AddEnemyChoices(state, choices, team, false); break;
            case 28: case 29: case 30: AddEnemyChoices(state, choices, team, true); break;
            case 31: case 32: case 33: case 34: case 35: AddFrozenChoices(state, choices, team); break;
            case 13: AddGenerationChoices(state, choices, team, PieceType.Cannon); break;
            case 14: AddGenerationChoices(state, choices, team, PieceType.Knight); break;
            case 15: AddRookPositionChoices(state, choices, team); break;
            case 17: case 18: case 19: case 20: case 21: AddReviveChoices(state, choices, team); break;
        }

        return choices;
    }

    public static void ResolveChoice(Gamestate state, int outcome, LotteryChoice choice)
    {
        int team = state.currentTeam;
        if (choice == null)
        {
            ResolveAutomatic(state, outcome, team);
            return;
        }

        switch (outcome)
        {
            case 1:
            case 2:
            case 3:
            case 4:
            case 6:
            case 7:
            case 8:
            case 12:
                UpgradeAt(state, choice.x, choice.y, team, UpgradeType(outcome)); break;
            case 11:
                UpgradeAt(state, choice.x, choice.y, team, 1);
                UpgradeAt(state, choice.secondX, choice.secondY, team, 1); break;
            case 16: DefectAt(state, choice.x, choice.y, team); break;
            case 28: case 29: case 30: FreezeAt(state, choice.x, choice.y, team); break;
            case 31: case 32: case 33: case 34: case 35: DefrostAt(state, choice.x, choice.y, team); break;
            case 13:
                GenerateAt(state, choice.x, choice.y, PieceType.Cannon, team);
                if (choice.secondX != int.MinValue) GenerateAt(state, choice.secondX, choice.secondY, PieceType.Cannon, team);
                break;
            case 14:
                GenerateAt(state, choice.x, choice.y, PieceType.Knight, team);
                if (choice.secondX != int.MinValue) GenerateAt(state, choice.secondX, choice.secondY, PieceType.Knight, team);
                break;
            case 15: GenerateAt(state, choice.x, choice.y, PieceType.Rook, team); break;
            case 17: case 18: case 19: case 20: case 21: ReviveAt(state, choice, team); break;
            default: ResolveAutomatic(state, outcome, team); break;
        }
    }

    private static void ResolveAutomatic(Gamestate state, int outcome, int team)
    {
        switch (outcome)
        {
            case 5: ResolveAutoUpgradeKing(state, team); break;
            case 9: PieceGenerator.GeneratePawnsOnRiver(state, team); break;
            case 10: PieceGenerator.GenerateWall(state, team); break;
            case 22: LotteryEffects.Reverse(state); break;
            case 23: LotteryEffects.Flood(state); break;
            case 24: LotteryEffects.ChargeBugle(state); break;
            case 25: LotteryEffects.LaserCannon(state); break;
            case 26: if (!state.isBoardExpanded) LotteryEffects.ExpandBoard(state); break;
            case 27: if (state.isBoardExpanded) LotteryEffects.ShrinkBoard(state); break;
        }
    }

    private static void UpgradeAt(Gamestate state, int x, int y, int team, int level)
    {
        Piece piece = state[x, y];
        if (piece.thisTeam == team && piece.upgradeLevel != level
            && piece.upgradeLevel + level <= MaxUpgradeLevel(piece.type))
            piece.Upgrade(piece.upgradeLevel + level);
    }

    private static void DefectAt(Gamestate state, int x, int y, int team)
    {
        Piece piece = state[x, y];
        if (piece.type != PieceType.Empty && piece.type != PieceType.Wall
            && piece.type != PieceType.King && piece.thisTeam == -team)
            piece.Defect();
    }

    private static void FreezeAt(Gamestate state, int x, int y, int team)
    {
        Piece piece = state[x, y];
        if (piece.type != PieceType.Empty && piece.type != PieceType.Wall && piece.thisTeam == -team)
            piece.frozenTurns = 6;
    }

    private static void DefrostAt(Gamestate state, int x, int y, int team)
    {
        Piece piece = state[x, y];
        if (piece.thisTeam == team && piece.frozenTurns > 0)
            piece.frozenTurns = 0;
    }

    private static void GenerateAt(Gamestate state, int x, int y, PieceType type, int team)
    {
        if (!state.IsValidPosition(x, y)) return;
        Piece target = state[x, y];
        if (target.type == PieceType.Wall
            || (target.type != PieceType.Empty && target.thisTeam == team && Piece.friendlyFire != 1)) return;
        if (target.type != PieceType.Empty && target.isDead)
            state.AddToGraveyard(target);
        PieceGenerator.PlacePieceAt(state, x, y, type);
    }

    private static void ReviveAt(Gamestate state, LotteryChoice choice, int team)
    {
        var graveyard = team == 1 ? state.redGraveyard : state.blackGraveyard;
        if (choice.graveyardIndex < 0 || choice.graveyardIndex >= graveyard.Count) return;
        Piece target = state[choice.x, choice.y];
        if (target.type == PieceType.Wall
            || (target.type != PieceType.Empty && target.thisTeam == team && Piece.friendlyFire != 1)) return;
        Piece piece = graveyard[choice.graveyardIndex];
        PieceGenerator.RevivePiece(state, piece, choice.x, choice.y);
        piece.frozenTurns = 0;
        if (piece is Pawn pawn && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3))
        {
            pawn.sniperCooldown = 2;
            pawn.sniperAvailable = false;
        }
    }

    private static int UpgradeType(int outcome) => outcome == 3 || outcome == 7 ? 2 : 1;

    private static void AddPieceChoices(Gamestate state, List<LotteryChoice> choices,
        PieceType type, int level, int team)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == type && p.thisTeam == team && p.upgradeLevel != level
                    && p.upgradeLevel + level <= MaxUpgradeLevel(type))
                    choices.Add(new LotteryChoice(LotteryChoiceType.Piece, x, y));
            }
    }

    private static void AddTwoPieceChoices(Gamestate state, List<LotteryChoice> choices,
        PieceType type, int level, int team)
    {
        var candidates = new List<(int x, int y)>();
        AddCandidatePositions(state, candidates, type, level, team);
        for (int i = 0; i < candidates.Count; i++)
            for (int j = i + 1; j < candidates.Count; j++)
                choices.Add(new LotteryChoice(LotteryChoiceType.TwoPieces,
                    candidates[i].x, candidates[i].y, candidates[j].x, candidates[j].y));
    }

    private static void AddCandidatePositions(Gamestate state, List<(int x, int y)> candidates,
        PieceType type, int level, int team)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == type && p.thisTeam == team && p.upgradeLevel != level
                    && p.upgradeLevel + level <= MaxUpgradeLevel(type))
                    candidates.Add((x, y));
            }
    }

    private static void AddEnemyChoices(Gamestate state, List<LotteryChoice> choices,
        int team, bool excludeKing)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type != PieceType.Empty && p.type != PieceType.Wall
                    && p.thisTeam == -team && (!excludeKing || p.type != PieceType.King))
                    choices.Add(new LotteryChoice(LotteryChoiceType.Piece, x, y));
            }
    }

    private static void AddFrozenChoices(Gamestate state, List<LotteryChoice> choices, int team)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
                if (state[x, y].thisTeam == team && state[x, y].frozenTurns > 0)
                    choices.Add(new LotteryChoice(LotteryChoiceType.Piece, x, y));
    }

    private static void AddGenerationChoices(Gamestate state, List<LotteryChoice> choices,
        int team, PieceType type)
    {
        var positions = new List<(int x, int y)>();
        if (type == PieceType.Cannon)
        {
            int[] dx = { 0, 0, 1, -1 };
            int[] dy = { 1, -1, 0, 0 };
            for (int x = state.leftBound; x <= state.rightBound; x++)
                for (int y = state.lowerBound; y <= state.upperBound; y++)
                {
                    if (state[x, y].type != PieceType.Pawn || state[x, y].thisTeam != team) continue;
                    for (int d = 0; d < 4; d++)
                    {
                        int nx = x + dx[d], ny = y + dy[d];
                        if (state.IsValidPosition(nx, ny) && state[nx, ny].type != PieceType.Wall
                            && !positions.Contains((nx, ny))) positions.Add((nx, ny));
                    }
                }
        }
        else
        {
            for (int x = state.leftBound; x <= state.rightBound; x++)
                for (int y = state.lowerBound; y <= state.upperBound; y++)
                    if (IsOnFriendlyBorder(x, y, team)) positions.Add((x, y));
        }

        positions.RemoveAll(pos =>
        {
            Piece target = state[pos.x, pos.y];
            return target.type == PieceType.Wall
                || (target.type != PieceType.Empty && target.thisTeam == team && Piece.friendlyFire != 1);
        });

        if (positions.Count == 1)
            choices.Add(new LotteryChoice(LotteryChoiceType.Position, positions[0].x, positions[0].y));
        else
            for (int i = 0; i < positions.Count; i++)
                for (int j = i + 1; j < positions.Count; j++)
                    choices.Add(new LotteryChoice(LotteryChoiceType.TwoPositions,
                        positions[i].x, positions[i].y, positions[j].x, positions[j].y));
    }

    private static void AddRookPositionChoices(Gamestate state, List<LotteryChoice> choices, int team)
    {
        foreach (var (x, y) in state.GetInitialPositions(PieceType.Rook, team))
        {
            Piece target = state[x, y];
            if (target.type != PieceType.Wall
                && (target.type == PieceType.Empty || target.thisTeam != team || Piece.friendlyFire == 1))
                choices.Add(new LotteryChoice(LotteryChoiceType.Position, x, y));
        }
    }

    private static void AddReviveChoices(Gamestate state, List<LotteryChoice> choices, int team)
    {
        var graveyard = team == 1 ? state.redGraveyard : state.blackGraveyard;
        for (int i = 0; i < graveyard.Count; i++)
            foreach (var (x, y) in state.GetInitialPositions(graveyard[i].type, team))
            {
                Piece target = state[x, y];
                if (target.type != PieceType.Wall
                    && (target.type == PieceType.Empty || target.thisTeam != team || Piece.friendlyFire == 1))
                    choices.Add(new LotteryChoice(LotteryChoiceType.Revive, x, y, graveyardIndex: i));
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
