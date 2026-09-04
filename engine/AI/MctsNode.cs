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
    public bool pruned;          // 永久剪枝：合法性失败时置位（棋子已移动/冻结等），PUCT 不再选择。
                                 // 注意：重复检测的临时跳过自 2026-09-04 起由引擎侧 worker 本地
                                 // 集合（simRepeatSkip）管理，不再写本字段，二者语义已分离。
    public Dictionary<string, double> priorByAction; // 叶节点缓存的所有合法动作先验（actionKey→prob）

    // ── ChanceNode 固化选择 ──
    // 仅 outcome 子节点使用：该 outcome 首次创建时实际执行的 LotteryChoice。
    // 重放时必须执行完全相同的 choice，保证子树状态一致性
    // （null = 无可选目标的自动效果路径）
    public LotteryChoice fixedChoice;

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
