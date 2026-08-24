using System;
using System.Collections.Generic;

// ====================================================================
// 统一行动抽象。MCTS 不区分走棋/狙击/连环马/抽奖，一视同仁。
// Execute 负责修改 state（含切换 currentTeam），调用方无需额外处理。
// ====================================================================

public abstract class GameAction
{
    public int team; // 执行此行动的阵营

    /// <summary>在给定 state 上执行此行动（会修改 state，含回合切换和状态维护）</summary>
    public abstract void Execute(Gamestate state, System.Random rng);

    public abstract string GetDescription();

    /// <summary>回合切换 + 冻结/狙击冷却更新 + 墙持续回合更新。
    /// 每个 Execute 末尾调用。准备模式（prepareModeOn）不更新冷却和墙。</summary>
    public static void EndTurn(Gamestate state)
    {
        state.currentTeam = -state.currentTeam;

        // 准备阶段：狙击不冷却、冻结不恢复、墙不减少
        if (state.prepareModeOn)
            return;

        state.UpdateFrozenTurns();
        state.UpdateAllSniperCooldowns();
        UpdateWalls(state);
    }

    /// <summary>墙持续回合递减，为 0 时消失（秦王绕柱）</summary>
    private static void UpdateWalls(Gamestate state)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p is Wall wall)
                {
                    wall.wallDuration--;
                    if (wall.wallDuration <= 0)
                        state[x, y] = Empty.Instance;
                }
            }
        }
    }
}

// ====================================================================
// 普通走棋
// ====================================================================
public class MoveAction : GameAction
{
    public int fromX, fromY, toX, toY;

    public MoveAction(int fx, int fy, int tx, int ty, int team)
    {
        fromX = fx; fromY = fy; toX = tx; toY = ty;
        this.team = team;
    }

    public override void Execute(Gamestate state, System.Random rng)
    {
        Piece piece = state[fromX, fromY];
        Piece target = state[toX, toY];

        // 防御：若棋子对象的坐标字段与棋盘位置不一致（抽奖效果可能留下脏数据），以棋盘位置为准
        if (piece.thisx != fromX || piece.thisy != fromY)
        {
            piece.thisx = fromX;
            piece.thisy = fromY;
        }

        piece.Move(toX, toY, state);

        if (target.type != PieceType.Empty && target.isDead)
            state.AddToGraveyard(target);

        EndTurn(state);
    }

    public override string GetDescription() =>
        $"Move ({fromX},{fromY})→({toX},{toY})";
}

// ====================================================================
// 狙击兵狙击
// ====================================================================
public class SniperAction : GameAction
{
    public int fromX, fromY, dx, dy;

    public SniperAction(int fx, int fy, int dx, int dy, int team)
    {
        fromX = fx; fromY = fy; this.dx = dx; this.dy = dy;
        this.team = team;
    }

    public override void Execute(Gamestate state, System.Random rng)
    {
        Pawn sniper = (Pawn)state[fromX, fromY];
        if (sniper.thisx != fromX || sniper.thisy != fromY)
        {
            sniper.thisx = fromX;
            sniper.thisy = fromY;
        }
        Piece target = sniper.GetSnipeTarget(dx, dy, state);

        sniper.Snipe(dx, dy, state);

        if (target != null && target.isDead)
            state.AddToGraveyard(target);

        EndTurn(state);
    }

    public override string GetDescription() =>
        $"Snipe ({fromX},{fromY})→dir({dx},{dy})";
}

// ====================================================================
// 抽奖（随机环境，抽奖结果随机；主动目标由调用方决定）
// ====================================================================
public class LotteryAction : GameAction
{
    public int lastOutcome;

    public LotteryAction(int team) { this.team = team; }

    public override void Execute(Gamestate state, System.Random rng)
    {
        lastOutcome = rng.Next(1, 41);
        LotteryResolver.Resolve(state, lastOutcome, rng);
        EndTurn(state);
    }

    public override string GetDescription() => "Lottery";
}

// ====================================================================
// 枚举当前 state 下某阵营所有合法行动
// ====================================================================
public static class ActionGenerator
{
    public static List<GameAction> GetAllActions(Gamestate state, int team)
    {
        var actions = new List<GameAction>();

        // 准备模式：只能抽奖（强制，即使 no-lottery AI 也必须抽）
        if (state.prepareModeOn)
        {
            actions.Add(new LotteryAction(team));
            return actions;
        }

        // 1. 普通走棋
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece piece = state[x, y];
                if (piece.type == PieceType.Empty || piece.thisTeam != team)
                    continue;
                if (piece.frozenTurns > 0)
                    continue;

                for (int tx = state.leftBound; tx <= state.rightBound; tx++)
                {
                    for (int ty = state.lowerBound; ty <= state.upperBound; ty++)
                    {
                        if (piece.IsLegalMove(tx, ty, state))
                            actions.Add(new MoveAction(x, y, tx, ty, team));
                    }
                }
            }
        }

        // 2. 狙击兵行动
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece piece = state[x, y];
                if (piece is Pawn pawn && pawn.thisTeam == team
                    && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3)
                    && pawn.sniperAvailable && pawn.frozenTurns == 0)
                {
                    // 上(0,1) 下(0,-1) 左(-1,0) 右(1,0)
                    TryAddSniper(actions, state, pawn, team, 0, 1);
                    TryAddSniper(actions, state, pawn, team, 0, -1);
                    TryAddSniper(actions, state, pawn, team, -1, 0);
                    TryAddSniper(actions, state, pawn, team, 1, 0);
                }
            }
        }

        // 3. 抽奖（永远可选）
        actions.Add(new LotteryAction(team));

        return actions;
    }

    private static void TryAddSniper(List<GameAction> actions, Gamestate state,
        Pawn pawn, int team, int dx, int dy)
    {
        if (pawn.CanSnipeInDirection(dx, dy, state))
            actions.Add(new SniperAction(pawn.thisx, pawn.thisy, dx, dy, team));
    }
}
