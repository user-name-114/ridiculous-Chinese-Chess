using System;
using System.Collections.Generic;

// ====================================================================
// 对弈引擎：AI vs AI / AI vs 随机，纯数据层，不依赖 Unity。
//
// 用法：
//   var result = MatchRunner.Run(state, redAI, blackAI, maxMoves: 300);
//   // result.Winner: 1=红胜, -1=黑胜, 0=和
//   // result.Steps: 每步的 (team, action, probabilityDistribution)
// ====================================================================

public class MatchStep
{
    public int team;                                    // 行动方 (1/-1)
    public GameAction action;                           // 采取的行动
    public List<(GameAction action, double prob)> distribution; // MCTS 根节点概率分布
}

public class MatchResult
{
    public int winner;             // 1=红胜, -1=黑胜, 0=和
    public int totalMoves;
    public string endReason;       // "king_captured" / "max_moves"
    public List<MatchStep> steps;  // 每步记录
    public int aiLotteryCount;     // AI 选择抽奖的次数
    public int redLotteryCount;    // 红方抽奖次数
    public int blackLotteryCount;  // 黑方抽奖次数
}

public static class MatchRunner
{
    /// <summary>AI vs AI 完整对局。prepareMode=true 时双方开局各抽 5 次奖。</summary>
    public static MatchResult Run(Gamestate initialState,
        AIPlayer red, AIPlayer black, int maxMoves = 300, bool recordSteps = true,
        bool prepareMode = false, int repetitionLoser = 0)
    {
        Gamestate state = initialState.DeepClone();
        var steps = recordSteps ? new List<MatchStep>() : null;
        int aiLotteryCount = 0;
        int redLottery = 0, blackLottery = 0;
        System.Random rng = new System.Random();

        // ── 准备阶段：双方交替各抽 5 次奖（共 10 轮）──
        // 期间不切换 currentTeam（抽奖本身不计为行动回合切换），
        // 由 EndTurnAndUpdate 风格的计数控制。
        // 狙击手冷却不恢复、冻结不自然解除、墙不减少。
        if (prepareMode)
        {
            state.prepareModeOn = true;
            for (int round = 0; round < 10; round++)
            {
                state.currentTeam = (round % 2 == 0) ? 1 : -1;
                int outcome = rng.Next(1, 41);
                LotteryResolver.Resolve(state, outcome, rng);
                if (state.currentTeam == 1) redLottery++;
                else blackLottery++;
                aiLotteryCount++;

                state.prepareLotteryCount++;
                if (state.prepareLotteryCount >= 10)
                    state.prepareModeOn = false;
            }
            // 准备结束，红方先手
            state.currentTeam = 1;
        }
        else
        {
            // 默认关闭准备模式（GameState.prepareModeOn 默认 true，需显式关闭）
            state.prepareModeOn = false;
        }

        var repetitionTracker = new RepetitionTracker(state);

        for (int moveCount = 0; moveCount < maxMoves; moveCount++)
        {
            int team = state.currentTeam;

            // 终局检查
            if (MctsEngine.IsTerminal(state))
            {
                int winner = -team; // 当前方被将死，对方胜
                return new MatchResult
                {
                    winner = winner,
                    totalMoves = moveCount,
                    endReason = "king_captured",
                    steps = steps,
                    aiLotteryCount = aiLotteryCount,
                    redLotteryCount = redLottery,
                    blackLotteryCount = blackLottery
                };
            }

            // 选择 AI
            AIPlayer ai = (team == 1) ? red : black;
            GameAction action = null;
            List<(GameAction, double)> dist = null;

            if (ai != null)
            {
                dist = ai.GetActionDistribution(state);
                // 取概率最大的行动
                double bestP = -1;
                foreach (var (a, p) in dist)
                {
                    if (p > bestP) { bestP = p; action = a; }
                }
            }
            else
            {
                action = RandomPick(state, team, rng);
            }

            if (action == null) break; // 不应发生

            // 统计抽奖
            if (action is LotteryAction)
            {
                if (ai != null) aiLotteryCount++;
                if (team == 1) redLottery++;
                else blackLottery++;
            }

            // 记录
            if (recordSteps)
            {
                steps.Add(new MatchStep
                {
                    team = team,
                    action = action,
                    distribution = dist
                });
            }

            // 执行
            if (ai != null)
                ai.ExecuteAction(state, action);
            else
                action.Execute(state, rng);

            // 最近 30 步内同一局面出现第 3 次，判"刚走的一方"负
            bool countRepetition = !(action is LotteryAction lottery
                && lottery.lastOutcome >= 36);
            if (repetitionTracker.AddState(state, countRepetition))
            {
                // 默认判"刚走的一方"负；若指定 repetitionLoser（评测时固定为 onnx），则固定判该方负
                int loser = repetitionLoser != 0 ? repetitionLoser : team;
                return new MatchResult
                {
                    winner = -loser,
                    totalMoves = moveCount + 1,
                    endReason = "重复局面",
                    steps = steps,
                    aiLotteryCount = aiLotteryCount,
                    redLotteryCount = redLottery,
                    blackLotteryCount = blackLottery
                };
            }
        }

        // 达到最大步数
        return new MatchResult
        {
            winner = 0,
            totalMoves = maxMoves,
            endReason = "max_moves",
            steps = steps,
            aiLotteryCount = aiLotteryCount
        };
    }

    /// <summary>AI（指定阵营）vs 随机对手</summary>
    public static MatchResult RunVsRandom(Gamestate initialState,
        AIPlayer ai, int aiTeam, int maxMoves = 400, bool prepareMode = false)
    {
        AIPlayer red = (aiTeam == 1) ? ai : null;
        AIPlayer black = (aiTeam == -1) ? ai : null;
        return Run(initialState, red, black, maxMoves, recordSteps: false,
            prepareMode: prepareMode);
    }

    /// <summary>随机 vs 随机，用于基准测试</summary>
    public static MatchResult RunRandomVsRandom(Gamestate initialState,
        int maxMoves = 300)
    {
        return Run(initialState, null, null, maxMoves, recordSteps: false);
    }

    // ================================================================
    //  随机行动选择
    // ================================================================
    private static GameAction RandomPick(Gamestate state, int team, System.Random rng)
    {
        var actions = ActionGenerator.GetAllActions(state, team);
        if (actions.Count == 0) return null;
        return actions[rng.Next(actions.Count)];
    }
}
