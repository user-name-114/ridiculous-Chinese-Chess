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
        double dirichletAlpha = 0, double dirichletEpsilon = 0)
    {
        Simulations = simulations;
        AllowLottery = allowLottery;
        this.aiTeam = aiTeam;
        engine = new MctsEngine(simulations, C, maxRolloutDepth: 200,
            allowLottery: allowLottery, aiTeam: aiTeam,
            lotteryCMultiplier: lotteryCMultiplier,
            useLotteryChanceNodes: useLotteryChanceNodes,
            threadCount: threadCount, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon);
        rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
    }

    /// <summary>给定棋盘状态，返回 AI 认为最优的行动</summary>
    public GameAction GetBestAction(Gamestate state)
    {
        return engine.FindBestAction(state, rng);
    }

    /// <summary>返回根节点各行动的概率分布（visitCount 归一化），用于自对弈数据收集</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(Gamestate state)
    {
        return engine.GetActionDistribution(state, rng);
    }

    /// <summary>重置随机种子（用于可复现测试）</summary>
    public void ResetRandom(int? seed = null)
    {
        rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();
    }
}
