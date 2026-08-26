using System;
using System.Collections.Generic;

// ====================================================================
// MCTS 搜索树节点。不存储 Gamestate，状态在遍历时动态生成。
// totalValue 是从执行此节点 action 的玩家视角累积的价值。
// ====================================================================
public class MctsNode
{
    public GameAction action;                  // 到达此节点的行动（根节点为 null）
    public MctsNode parent;
    public List<MctsNode> children = new List<MctsNode>();
    public int visitCount;
    public double totalValue;
    public double prior;         // 先验概率 P(s,a)，来自策略网络 softmax（纯 MCTS 时为均匀）
    public bool fullyExpanded;   // 是否已完全展开（所有合法行动都已创建子节点）
    public bool pruned;          // 被重复检测标记为非法（PUCT 不再选择此节点）
    public Dictionary<string, double> priorByAction; // 叶节点缓存的所有合法动作先验（actionKey→prob）

    // ── Scheme B: Chance Node 支持 ──
    public bool IsChanceNode;                  // 此节点是否代表随机事件（抽奖）
    public Dictionary<int, MctsNode> outcomeChildren; // outcome → child (lazy)
    public int sampledOutcome;                 // 最近一次采样到的抽奖结果编号

    public MctsNode(GameAction action, MctsNode parent)
    {
        this.action = action;
        this.parent = parent;
    }

    /// <summary>
    /// PUCT 值（AlphaZero）：Q + C * prior * sqrt(N_parent) / (1 + N)。
    /// 未访问时 Q=0，完全靠先验 prior 引导探索，不再返回无穷大。
    /// </summary>
    public double PuctValue(double C, int parentVisits)
    {
        double q = visitCount == 0 ? 0.0 : (totalValue / visitCount);
        double u = C * prior * Math.Sqrt(parentVisits) / (1.0 + visitCount);
        return q + u;
    }
}
