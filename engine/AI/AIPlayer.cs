using System;
using System.Collections.Generic;

// ====================================================================
// AI 玩家：对外唯一接口。
//
// 用法（Unity 中）：
//   AIPlayer ai = new AIPlayer(simulations: 500);
//   GameAction best = ai.GetBestAction(state);
//   // best 是 MoveAction / LotteryAction / SniperAction 之一
//
// 自对弈数据收集：
//   var dist = ai.GetActionDistribution(state);
//   // dist 是 (action, probability) 列表，probability 由 visitCount 归一化
// ====================================================================
public class AIPlayer
{
    private MctsEngine engine;
    private System.Random rng;
    private int aiTeam;

    public int Simulations { get; }
    public bool AllowLottery { get; }

    public AIPlayer(int simulations = 500, double C = 1.2, int? seed = null,
        bool allowLottery = true, int aiTeam = 0,
        double lotteryCMultiplier = 1.0, bool useLotteryChanceNodes = true,
        int threadCount = 16, NeuralMcts neural = null,
        double dirichletAlpha = 0, double dirichletEpsilon = 0,
        double evalMaterialWeight = 0.15,
        double virtualLossValue = 0.5,
        int lotteryEvalLimit = 16,
        int maxRolloutDepth = 200)
    {
        Simulations = simulations;
        AllowLottery = allowLottery;
        this.aiTeam = aiTeam;
        engine = new MctsEngine(simulations, C, maxRolloutDepth: maxRolloutDepth,
            allowLottery: allowLottery, aiTeam: aiTeam,
            lotteryCMultiplier: lotteryCMultiplier,
            useLotteryChanceNodes: useLotteryChanceNodes,
            threadCount: threadCount, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon,
            evalMaterialWeight: evalMaterialWeight,
            virtualLossValue: virtualLossValue,
            lotteryEvalLimit: lotteryEvalLimit);
        rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
    }

    /// <summary>给定棋盘状态，返回 AI 认为最优的行动</summary>
    public GameAction GetBestAction(Gamestate state)
    {
        return engine.FindBestAction(state, rng);
    }

    /// <summary>带真实对局历史，返回最优行动（让 MCTS 感知重复局面）</summary>
    public GameAction GetBestAction(Gamestate state, RepetitionTracker history)
    {
        return engine.FindBestAction(state, rng, history);
    }

    /// <summary>返回根节点各行动的概率分布（visitCount 归一化），用于自对弈数据收集</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(Gamestate state)
    {
        return engine.GetActionDistribution(state, rng);
    }

    /// <summary>带真实对局历史的概率分布（让 MCTS 感知重复局面，避免推荐会导致判负的走法）</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(
        Gamestate state, RepetitionTracker history)
    {
        return engine.GetActionDistribution(state, rng, history);
    }

    /// <summary>【诊断专用】根节点每个子动作的统计（含被剪枝项）</summary>
    public List<(GameAction action, string desc, bool isChance,
                 double prior, int visits, double q, bool pruned)> GetRootChildStats(
        Gamestate state, RepetitionTracker history)
    {
        return engine.GetRootChildStats(state, rng, history);
    }

    public void ExecuteAction(Gamestate state, GameAction action)
    {
        engine.ExecuteAction(state, action, rng);
    }

    /// <summary>重置随机种子（用于可复现测试）</summary>
    public void ResetRandom(int? seed = null)
    {
        rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
    }
}
