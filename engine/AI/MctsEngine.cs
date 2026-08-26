using System;
using System.Collections.Generic;
using System.Threading.Tasks;

// ====================================================================
// 支持两种抽奖处理方式：
//   Scheme A — lotteryCMultiplier > 1: 抽奖在 UCB 中获得更高探索权重
//   Scheme B — useLotteryChanceNodes: 抽奖作为 ChanceNode，
//             随机采样 outcome，渐进展开（Progressive Widening）
//
// 并行化：
//   threadCount > 1 时启用根并行化——多线程各跑独立树，最后合并。
//   推荐设为 CPU 逻辑核心数，默认 16。
//
// 用法：
//   var engine = new MctsEngine(simulations: 1000, C: 1.2,
//       useLotteryChanceNodes: true, threadCount: 16);
//   GameAction best = engine.FindBestAction(state, rng);
// ====================================================================
public sealed class RepetitionTracker
{
    public const int WindowSize = 20;
    private Dictionary<long, int> counts = new Dictionary<long, int>();
    private Queue<long> recent = new Queue<long>();
    private int ply;

    public RepetitionTracker()
    {
    }

    public RepetitionTracker(Gamestate state)
    {
        AddState(state);
    }

    /// <summary>创建一份快照（深拷贝 counts 和 recent），供 MCTS 模拟继承真实对局历史</summary>
    public RepetitionTracker Clone()
    {
        var clone = new RepetitionTracker();
        clone.ply = ply;
        foreach (var kv in counts)
            clone.counts[kv.Key] = kv.Value;
        foreach (var item in recent)
            clone.recent.Enqueue(item);
        return clone;
    }

    /// <summary>添加一个状态，返回该状态在窗口内是否已出现 3 次（第 3 次时返回 true）。
    /// isLottery=true 时不计入重复检测（抽奖不可控，不应因运气不好判负）</summary>
    public bool AddState(Gamestate state, bool isLottery = false)
    {
        long key = MctsEngine.RepetitionKey(state);
        if (!isLottery)
            AddKey(key);
        else
            // 抽奖仍滑动窗口（移除过期条目），但不增加计数
            AddKeyLottery(key);
        return !isLottery && counts.TryGetValue(key, out int c) && c >= 3;
    }

    /// <summary>检查如果添加这个状态，是否会构成第 3 次重复（不实际添加）</summary>
    public bool WouldRepeat(Gamestate state)
    {
        long key = MctsEngine.RepetitionKey(state);
        return counts.TryGetValue(key, out int c) && c >= 2;
    }

    private void AddKey(long key)
    {
        ply++;
        recent.Enqueue(key);
        counts.TryGetValue(key, out int c);
        counts[key] = c + 1;

        while (recent.Count > WindowSize)
        {
            var oldKey = recent.Dequeue();
            counts[oldKey]--;
            if (counts[oldKey] == 0)
                counts.Remove(oldKey);
        }
    }

    /// <summary>抽奖走法：滑动窗口但只标记"占位"，不增加重复计数</summary>
    private void AddKeyLottery(long key)
    {
        // 用负数标记抽奖占位，和正常计数的正数区分
        long placeholder = ~key; // 按位取反作为占位键
        ply++;
        recent.Enqueue(placeholder);
        // 不增加 counts[key]

        while (recent.Count > WindowSize)
        {
            var oldKey = recent.Dequeue();
            // 正常计数的 key 才需要递减
            if (oldKey >= 0 && counts.ContainsKey(oldKey))
            {
                counts[oldKey]--;
                if (counts[oldKey] == 0)
                    counts.Remove(oldKey);
            }
        }
    }
}

public class MctsEngine
{
    private int maxSimulations;
    private double explorationConstant;
    private int maxRolloutDepth;
    private bool allowLottery;
    private int aiTeam;
    private double lotteryCMultiplier;
    private bool useLotteryChanceNodes;
    private int threadCount;
    private NeuralMcts neural;
    private double dirichletAlpha;    // Dirichlet 噪声浓度（>0 时根节点加噪声，鼓励探索）
    private double dirichletEpsilon;  // 噪声混合比例

    public MctsEngine(int simulations = 1000, double C = 1.2, int maxRolloutDepth = 200,
        bool allowLottery = true, int aiTeam = 0,
        double lotteryCMultiplier = 1.0, bool useLotteryChanceNodes = false,
        int threadCount = 1, NeuralMcts neural = null,
        double dirichletAlpha = 0, double dirichletEpsilon = 0)
    {
        this.maxSimulations = simulations;
        this.explorationConstant = C;
        this.maxRolloutDepth = maxRolloutDepth;
        this.allowLottery = allowLottery;
        this.aiTeam = aiTeam;
        this.lotteryCMultiplier = lotteryCMultiplier;
        this.useLotteryChanceNodes = useLotteryChanceNodes;
        this.threadCount = Math.Max(1, threadCount);
        this.neural = neural;
        this.dirichletAlpha = dirichletAlpha;
        this.dirichletEpsilon = dirichletEpsilon;
    }

    // ================================================================
    //  公共接口
    // ================================================================

    /// <summary>在给定 state 上运行 MCTS，返回最优行动（visitCount 最大）</summary>
    public GameAction FindBestAction(Gamestate state, System.Random rng)
    {
        MctsNode root = RunMcts(state, rng);
        return BestChild(root);
    }

    /// <summary>运行 MCTS，返回根节点各行动的概率分布（visitCount 归一化）</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(
        Gamestate state, System.Random rng)
    {
        MctsNode root = RunMcts(state, rng);
        var dist = new List<(GameAction action, double probability)>();
        double total = root.visitCount;
        foreach (MctsNode child in root.children)
            dist.Add((child.action, child.visitCount / total));
        return dist;
    }

    /// <summary>运行 MCTS（带真实对局历史），返回概率分布</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(
        Gamestate state, System.Random rng, RepetitionTracker history)
    {
        MctsNode root;
        if (threadCount <= 1)
            root = RunMctsSingle(state, rng, null, history);
        else
            root = RunMcts(state, rng); // 根并行化模式不支持历史传入
        var dist = new List<(GameAction action, double probability)>();
        double total = root.visitCount;
        if (total == 0)
        {
            // 所有模拟都被剪枝，回退到无历史搜索
            return GetActionDistribution(state, rng);
        }
        foreach (MctsNode child in root.children)
            dist.Add((child.action, child.visitCount / total));
        return dist;
    }

    public void ExecuteAction(Gamestate state, GameAction action, System.Random rng)
    {
        if (action is LotteryAction)
        {
            var lottery = (LotteryAction)action;
            lottery.lastOutcome = rng.Next(1, 41);
            ExecuteLotteryOutcome(state, lottery.lastOutcome, rng);
            GameAction.EndTurn(state);
        }
        else
            action.Execute(state, rng);
    }

    // ================================================================
    //  MCTS 主循环
    // ================================================================

    /// <summary>
    /// MCTS 主循环。
    /// threadCount==1: 单线程顺序执行。
    /// threadCount>1:  根并行化——多线程各跑独立树，最后合并根的子节点统计数据。
    /// </summary>
    private MctsNode RunMcts(Gamestate state, System.Random rng)
    {
        if (threadCount <= 1)
            return RunMctsSingle(state, rng);

        // ── 根并行化 ──
        int threads = Math.Min(threadCount, maxSimulations);
        int simsPerThread = maxSimulations / threads;
        int remainder = maxSimulations % threads;
        var partialRoots = new MctsNode[threads];

        Parallel.For(0, threads, t =>
        {
            int sims = simsPerThread + (t < remainder ? 1 : 0);
            var localRng = new System.Random(rng.Next());
            partialRoots[t] = RunMctsSingle(state, localRng, sims);
        });

        // ── 合并根节点的子节点 ──
        var mergedRoot = new MctsNode(null, null);
        var childIndex = new Dictionary<string, (MctsNode child, int visits, double value)>();

        foreach (var root in partialRoots)
        {
            mergedRoot.visitCount += root.visitCount;
            foreach (var child in root.children)
            {
                string key = ActionKey(child.action);
                if (!childIndex.TryGetValue(key, out var entry))
                    childIndex[key] = (child, 0, 0);
                entry = childIndex[key];
                childIndex[key] = (child, entry.visits + child.visitCount,
                                   entry.value + child.totalValue);
            }
        }

        foreach (var (key, entry) in childIndex)
        {
            var mc = new MctsNode(entry.child.action, mergedRoot);
            mc.visitCount = entry.visits;
            mc.totalValue = entry.value;
            mergedRoot.children.Add(mc);
        }

        return mergedRoot;
    }

    /// <summary>单线程 MCTS（可指定模拟次数，供并行版调用）</summary>
    private MctsNode RunMctsSingle(Gamestate state, System.Random rng, int? simsOverride = null)
    {
        return RunMctsSingle(state, rng, simsOverride, null);
    }

    /// <summary>单线程 MCTS，可传入真实对局历史用于重复检测</summary>
    private MctsNode RunMctsSingle(Gamestate state, System.Random rng, int? simsOverride,
        RepetitionTracker realHistory)
    {
        int sims = simsOverride ?? maxSimulations;
        MctsNode root = new MctsNode(null, null);

        int validSims = 0;
        int attempts = 0;
        int maxAttempts = sims * 3; // 安全阀：防止极端情况下无限循环
        while (validSims < sims && attempts < maxAttempts)
        {
            attempts++;
            Gamestate workState;
            try { workState = state.DeepClone(); }
            catch (Exception ex)
            { throw new Exception($"DeepClone failed at sim {validSims}: {ex.Message}", ex); }

            // 每次模拟克隆一份真实历史，从根节点重新遍历
            RepetitionTracker simTracker = realHistory?.Clone() ?? new RepetitionTracker();

            MctsNode leaf;
            try { leaf = Select(root, workState, rng, simTracker); }
            catch (Exception ex)
            { throw new Exception($"Select failed at sim {validSims}: {ex.Message}", ex); }

            // leaf == null 表示该模拟因所有走法被剪枝而无法继续，不计入有效次数
            if (leaf == null)
                continue;

            validSims++;

            double result;
            float[] priors = null;
            try
            {
                if (IsTerminal(workState))
                    result = Evaluate(workState);
                else if (neural != null)
                {
                    var (p, v) = neural.PredictBlocking(workState);
                    priors = p;
                    result = v;
                    // 网络 value 是固定红方视角，转成叶节点玩家视角（黑方时取反）
                    if (workState.currentTeam == -1)
                        result = -result;
                }
                else
                    result = Simulate(workState, rng);
            }
            catch (Exception ex)
            { throw new Exception($"Simulate failed at sim {validSims}: {ex.Message}", ex); }

            // 展开叶节点（策略网络先验或均匀先验）；ChanceNode 不在此展开（由 HandleChanceNode 管理）
            if (!IsTerminal(workState) && leaf.children.Count == 0 && !leaf.IsChanceNode)
            {
                ExpandAll(leaf, workState, priors);
                // 根节点加 Dirichlet 噪声（AlphaZero 探索，仅自对弈时 dirichletAlpha>0）
                if (leaf == root && dirichletAlpha > 0 && dirichletEpsilon > 0)
                    AddDirichletNoise(root, rng);
            }

            Backpropagate(leaf, result);
        }

        return root;
    }

    // ================================================================
    //  Select — 沿树向下遍历直到叶节点
    //
    //  重复检测采用剪枝而非惩罚：
    //    执行一个非抽奖动作后，检查新局面是否构成第 3 次重复。
    //    如果是，标记该子节点为 pruned（PUCT 不再选择），回退 state，
    //    重新选择其他子节点。所有子节点都被剪枝时返回 null（跳过本次模拟）。
    //    抽奖动作（含未中奖和无效抽奖）豁免剪枝——抽不出好结果无可厚非。
    // ================================================================

    private MctsNode Select(MctsNode node, Gamestate state, System.Random rng,
        RepetitionTracker repetitionTracker)
    {
        while (!IsTerminal(state))
        {
            // ── ChanceNode: 随机采样 outcome，创建子节点，继续深入 ──
            if (node.IsChanceNode)
            {
                MctsNode result = HandleChanceNode(node, state, rng, repetitionTracker);
                if (result == node)
                    return node;          // PW 限制达到，叶节点
                node = result;            // outcome 子节点，继续循环
                continue;
            }

            // ── 未展开（叶节点）→ 返回，交给 RunMctsSingle 展开 + 评估 ──
            if (node.children.Count == 0)
                return node;

            // ── PUCT 选择最优子节点（跳过 pruned）──
            MctsNode child = BestPuctChild(node);

            // 所有子节点都被剪枝或无合法子节点
            if (child == null)
                return null;

            // ── 抽奖子节点：交给 HandleChanceNode 处理，不在此执行 ──
            if (child.IsChanceNode)
            {
                node = child;
                continue;
            }

            // ── 普通节点：校验合法性 ──
            if (!IsActionValid(child.action, state))
            {
                child.pruned = true;
                continue;
            }

            // ── 执行动作，检查重复 ──
            // 先在 workState 的快照上试走，检查是否重复
            Gamestate testState = state.DeepClone();
            ExecuteAction(testState, child.action, rng);

            // 抽奖动作豁免重复检测
            bool isLottery = child.action is LotteryAction;
            if (!isLottery && repetitionTracker.WouldRepeat(testState))
            {
                // 标记此走法为非法，重新选择
                child.pruned = true;
                continue;
            }

            // 确认执行：推进真实 state 和 tracker
            ExecuteAction(state, child.action, rng);
            repetitionTracker.AddState(state);

            node = child;
        }

        return node; // 终局
    }

    // ================================================================
    //  HandleChanceNode — Scheme B 抽奖随机节点
    //
    //  均匀随机采样 outcome (1~40)，执行效果，推进 state。
    //  使用 Progressive Widening 创建 outcome 子节点：
    //    最大子节点数 ≈ sqrt(visitCount) + 3
    //  返回 outcome 子节点（继续深入）或原节点（PW 限制，叶节点）。
    // ================================================================

    private MctsNode HandleChanceNode(MctsNode node, Gamestate state, System.Random rng,
        RepetitionTracker repetitionTracker)
    {
        int outcome = rng.Next(1, 41);
        node.sampledOutcome = outcome;
        ((LotteryAction)node.action).lastOutcome = outcome;

        ExecuteLotteryOutcome(state, outcome, rng);
        GameAction.EndTurn(state);

        // 抽奖豁免重复检测——不剪枝、不判负
        repetitionTracker.AddState(state, isLottery: true);

        // 查找已有 outcome 子节点
        if (node.outcomeChildren == null)
            node.outcomeChildren = new Dictionary<int, MctsNode>();

        if (node.outcomeChildren.TryGetValue(outcome, out MctsNode oc))
            return oc; // 已有子节点 → 继续深入

        // Progressive Widening
        int maxOC = (int)Math.Sqrt(node.visitCount + 1) + 3;
        if (node.outcomeChildren.Count < maxOC)
        {
            oc = new MctsNode(null, node); // outcome 子节点无 action
            node.outcomeChildren[outcome] = oc;
            return oc;
        }

        return node; // PW 限制达到
    }

    private void ExecuteLottery(Gamestate state, System.Random rng)
    {
        int outcome = rng.Next(1, 41);
        ExecuteLotteryOutcome(state, outcome, rng);
        GameAction.EndTurn(state);
    }

    private void ExecuteLotteryOutcome(Gamestate state, int outcome, System.Random rng)
    {
        List<LotteryChoice> choices = LotteryResolver.GetChoices(state, outcome);
        if (choices.Count == 0)
        {
            LotteryResolver.ResolveChoice(state, outcome, null);
            return;
        }

        LotteryChoice selected = neural == null
            ? choices[rng.Next(choices.Count)]
            : SelectLotteryChoiceByValue(state, outcome, choices);
        LotteryResolver.ResolveChoice(state, outcome, selected);
    }

    private LotteryChoice SelectLotteryChoiceByValue(Gamestate state, int outcome,
        List<LotteryChoice> choices)
    {
        LotteryChoice bestChoice = choices[0];
        double bestOpponentValue = double.PositiveInfinity;
        var candidates = new List<Gamestate>(choices.Count);
        var valueStates = new List<Gamestate>();
        var valueIndices = new List<int>();

        for (int i = 0; i < choices.Count; i++)
        {
            LotteryChoice choice = choices[i];
            Gamestate candidate = state.DeepClone();
            LotteryResolver.ResolveChoice(candidate, outcome, choice);
            GameAction.EndTurn(candidate);
            candidates.Add(candidate);

            if (IsTerminal(candidate))
                continue;

            valueIndices.Add(i);
            valueStates.Add(candidate);
        }

        float[] predictedValues = valueStates.Count == 0
            ? Array.Empty<float>()
            : neural.PredictValues(valueStates);

        int valueIndex = 0;
        for (int i = 0; i < candidates.Count; i++)
        {
            double opponentValue;
            if (IsTerminal(candidates[i]))
                opponentValue = Evaluate(candidates[i]);
            else
            {
                float predictedValue = predictedValues[valueIndex++];
                opponentValue = candidates[i].currentTeam == 1
                    ? predictedValue : -predictedValue;
            }

            if (opponentValue < bestOpponentValue)
            {
                bestOpponentValue = opponentValue;
                bestChoice = choices[i];
            }
        }

        return bestChoice;
    }

    // ================================================================
    //  Expand — 从当前状态未尝试的行动中随机取一个创建子节点
    //
    //  Scheme B 下 LotteryAction 创建为 ChanceNode，不在此执行
    //  （由 HandleChanceNode 负责执行）。
    // ================================================================

    /// <summary>
    /// 一次性展开叶节点的所有合法行动（AlphaZero 式），每个子节点用策略网络
    /// 的先验概率 P(s,a) 初始化（纯 MCTS 时用均匀先验）。动作在此不执行，
    /// 由 Select 在遍历时执行。
    /// </summary>
    private void ExpandAll(MctsNode node, Gamestate state, float[] rootPolicy)
    {
        var allActions = GetFilteredActions(state);
        int n = allActions.Count;
        if (n == 0) return;

        double[] probs = new double[n];
        if (rootPolicy != null)
        {
            double maxLogit = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                int idx = ActionToIndex(allActions[i]);
                double lg = (idx >= 0 && idx < rootPolicy.Length) ? rootPolicy[idx] : double.NegativeInfinity;
                probs[i] = lg;
                if (lg > maxLogit) maxLogit = lg;
            }
            double sum = 0;
            for (int i = 0; i < n; i++) { probs[i] = Math.Exp(probs[i] - maxLogit); sum += probs[i]; }
            if (sum > 0) for (int i = 0; i < n; i++) probs[i] /= sum;
            else for (int i = 0; i < n; i++) probs[i] = 1.0 / n;
        }
        else
        {
            for (int i = 0; i < n; i++) probs[i] = 1.0 / n;
        }

        for (int i = 0; i < n; i++)
        {
            var child = new MctsNode(allActions[i], node) { prior = probs[i] };
            node.children.Add(child);
            if (useLotteryChanceNodes && allActions[i] is LotteryAction)
                child.IsChanceNode = true;
        }
    }

    /// <summary>根节点先验加 Dirichlet 噪声：P' = (1-ε)P + ε·Dir(α)，鼓励探索</summary>
    private void AddDirichletNoise(MctsNode root, System.Random rng)
    {
        int n = root.children.Count;
        if (n == 0) return;
        double[] noise = SampleDirichlet(dirichletAlpha, n, rng);
        for (int i = 0; i < n; i++)
            root.children[i].prior = (1 - dirichletEpsilon) * root.children[i].prior
                                     + dirichletEpsilon * noise[i];
    }

    /// <summary>采样 Dirichlet(α, ..., α) 分布（n 个分量，和为 1）</summary>
    private static double[] SampleDirichlet(double alpha, int n, System.Random rng)
    {
        double[] g = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            g[i] = GammaSample(alpha, rng);
            sum += g[i];
        }
        for (int i = 0; i < n; i++) g[i] /= sum;
        return g;
    }

    /// <summary>采样 Gamma(shape=α, scale=1)，支持 α&lt;1（Dirichlet 常用 α&lt;1）</summary>
    private static double GammaSample(double alpha, System.Random rng)
    {
        if (alpha >= 1.0)
            return MarsagliaTsangGamma(alpha, rng);
        // α < 1: Gamma(α) = Gamma(α+1) · U^(1/α)
        double u = rng.NextDouble();
        if (u <= 0) u = 1e-12;
        return MarsagliaTsangGamma(alpha + 1.0, rng) * Math.Pow(u, 1.0 / alpha);
    }

    /// <summary>Marsaglia-Tsang 算法采样 Gamma(shape=α≥1, scale=1)</summary>
    private static double MarsagliaTsangGamma(double alpha, System.Random rng)
    {
        double d = alpha - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double z = StandardNormal(rng);
            double v = 1.0 + c * z;
            v = v * v * v;
            if (v <= 0) continue;
            double u = rng.NextDouble();
            if (u <= 0) continue;
            if (Math.Log(u) < 0.5 * z * z + d - d * v + d * Math.Log(v))
                return d * v;
        }
    }

    /// <summary>Box-Muller 采样标准正态分布</summary>
    private static double StandardNormal(System.Random rng)
    {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        if (u1 <= 0) u1 = 1e-12;
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // ================================================================
    //  Simulate — 从当前状态随机走子到终局（rollout）
    // ================================================================

    private double Simulate(Gamestate state, System.Random rng)
    {
        for (int depth = 0; depth < maxRolloutDepth; depth++)
        {
            if (IsTerminal(state))
                return Evaluate(state);

            // 轻量随机走子（不创建 GameAction 对象，避免 GC 压力）
            if (!LightRandomMove(state, rng))
                break;
        }

        return Evaluate(state);
    }

    /// <summary>
    /// 轻量随机走子：随机选一个己方棋子，随机选它的一个合法走法，直接执行。
    /// 不枚举所有走法、不创建 GameAction 对象，避免海量短命对象导致 GC 串行化。
    /// 只走普通棋（忽略抽奖/狙击，rollout 纯随机近似足够）。
    /// </summary>
    private bool LightRandomMove(Gamestate state, System.Random rng)
    {
        int team = state.currentTeam;
        int xMin = state.leftBound, xMax = state.rightBound;
        int yMin = state.lowerBound, yMax = state.upperBound;

        // 随机选一个己方棋子（最多尝试 8 次）
        for (int attempt = 0; attempt < 8; attempt++)
        {
            int px = rng.Next(xMin, xMax + 1);
            int py = rng.Next(yMin, yMax + 1);
            Piece piece = state[px, py];
            if (piece.type == PieceType.Empty || piece.thisTeam != team || piece.frozenTurns > 0)
                continue;

            // 随机选一个合法目标（最多尝试 8 次）
            for (int attempt2 = 0; attempt2 < 8; attempt2++)
            {
                int tx = rng.Next(xMin, xMax + 1);
                int ty = rng.Next(yMin, yMax + 1);
                if (!piece.IsLegalMove(tx, ty, state))
                    continue;

                Piece target = state[tx, ty];
                piece.Move(tx, ty, state);
                if (target.type != PieceType.Empty && target.isDead)
                    state.AddToGraveyard(target);
                GameAction.EndTurn(state);
                return true;
            }
        }
        return false; // 无合法走子
    }

    // ================================================================
    //  Backpropagate — 模拟结果沿路径回传
    //
    //  零和博弈：经过一个已执行的动作节点时翻转 value 视角。
    //  outcome 子节点没有 action，不额外翻转；其父 ChanceNode 代表抽奖动作，
    //  会像普通动作一样完成一次视角转换。
    // ================================================================

    private void Backpropagate(MctsNode node, double result)
    {
        while (node != null)
        {
            // 叶节点 value 属于动作执行后的当前行动方；动作节点需要先转换
            // 回执行该动作一方的视角，再写入自身统计值。
            if (node.action != null)
                result = -result;

            node.visitCount++;
            node.totalValue += result;

            node = node.parent;
        }
    }

    // ================================================================
    //  辅助方法
    // ================================================================

    private List<GameAction> GetFilteredActions(Gamestate state)
    {
        var actions = ActionGenerator.GetAllActions(state, state.currentTeam);
        // 准备模式下不检查 allowLottery（强制抽奖）
        if (!allowLottery && state.currentTeam == aiTeam && !state.prepareModeOn)
            actions.RemoveAll(a => a is LotteryAction);
        return actions;
    }

    private static bool IsActionValid(GameAction action, Gamestate state)
    {
        if (action is MoveAction m)
        {
            Piece piece = state[m.fromX, m.fromY];
            if (piece.type == PieceType.Empty || piece.thisTeam != action.team)
                return false;
            if (piece.frozenTurns > 0) return false;
            return piece.IsLegalMove(m.toX, m.toY, state);
        }
        if (action is SniperAction s)
        {
            Piece piece = state[s.fromX, s.fromY];
            if (!(piece is Pawn pawn) || pawn.thisTeam != action.team)
                return false;
            return pawn.sniperAvailable && pawn.frozenTurns == 0
                && pawn.CanSnipeInDirection(s.dx, s.dy, state);
        }
        return true; // LotteryAction 永远合法
    }

    private static string ActionKey(GameAction a)
    {
        if (a is MoveAction m)
            return $"M{m.fromX},{m.fromY}>{m.toX},{m.toY}";
        if (a is SniperAction s)
            return $"S{s.fromX},{s.fromY}>{s.dx},{s.dy}";
        if (a is LotteryAction)
            return "L";
        return a.GetDescription();
    }

    /// <summary>把 GameAction 映射到策略头输出的根节点 logits 索引（0~24332）</summary>
    private static int ActionToIndex(GameAction a)
    {
        if (a is MoveAction m)
            return ActionEncoder.EncodeMove(m.fromX, m.fromY, m.toX, m.toY);
        if (a is SniperAction s)
            return ActionEncoder.EncodeSniper(s.fromX, s.fromY, s.dx, s.dy);
        if (a is LotteryAction)
            return StateEncoder.MoveActionSize + StateEncoder.SniperActionSize; // 抽奖标量
        return -1;
    }

    // ================================================================
    //  终局判断 & 局面评估
    // ================================================================

    public static bool IsTerminal(Gamestate state)
    {
        return !HasKing(state, 1) || !HasKing(state, -1);
    }

    /// <summary>从当前行动方视角评估: +1 胜 / -1 负 / 0 其他</summary>
    public static double Evaluate(Gamestate state)
    {
        bool currentKing = HasKing(state, state.currentTeam);
        bool otherKing = HasKing(state, -state.currentTeam);

        if (!currentKing && otherKing) return -1;
        if (!otherKing && currentKing) return +1;
        return 0;
    }

    public static bool HasKing(Gamestate state, int team)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
                if (state[x, y] is King king && king.thisTeam == team)
                    return true;
        return false;
    }

    /// <summary>完整局面哈希，用于重复检测</summary>
    public static long StateHash(Gamestate state)
    {
        unchecked
        {
            long hash = 1469598103934665603L;
            AddHash(ref hash, state.leftBound);
            AddHash(ref hash, state.rightBound);
            AddHash(ref hash, state.lowerBound);
            AddHash(ref hash, state.upperBound);
            AddHash(ref hash, state.isBoardExpanded);
            AddHash(ref hash, state.prepareModeOn);
            AddHash(ref hash, state.prepareLotteryCount);
            AddHash(ref hash, state.currentTeam);
            AddHash(ref hash, state.lianHuanMaTeam);

            for (int i = 0; i < state.lianHuanMaTargets.Count; i++)
            {
                AddHash(ref hash, state.lianHuanMaTargets[i].x);
                AddHash(ref hash, state.lianHuanMaTargets[i].y);
            }

            AddBoardHash(ref hash, state);
            AddGraveyardHash(ref hash, state.redGraveyard);
            AddGraveyardHash(ref hash, state.blackGraveyard);
            return hash;
        }
    }

    private static void AddBoardHash(ref long hash, Gamestate state)
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                AddPieceHash(ref hash, p);
            }
    }

    private static void AddGraveyardHash(ref long hash, List<Piece> graveyard)
    {
        AddHash(ref hash, graveyard.Count);
        foreach (Piece piece in graveyard)
            AddPieceHash(ref hash, piece);
    }

    private static void AddPieceHash(ref long hash, Piece piece)
    {
        AddHash(ref hash, (int)piece.type);
        AddHash(ref hash, piece.thisTeam);
        AddHash(ref hash, piece.isDead);
        AddHash(ref hash, piece.thisx);
        AddHash(ref hash, piece.thisy);
        AddHash(ref hash, piece.upgradeLevel);
        AddHash(ref hash, piece.frozenTurns);
        AddHash(ref hash, piece.freezeTickCount);
        if (piece is Pawn pawn)
        {
            AddHash(ref hash, pawn.sniperCooldown);
            AddHash(ref hash, pawn.sniperAvailable);
        }
        if (piece is Wall wall)
            AddHash(ref hash, wall.wallDuration);
    }

    private static void AddHash(ref long hash, int value)
    {
        hash ^= (uint)value;
        hash *= 1099511628211L;
    }

    private static void AddHash(ref long hash, bool value)
    {
        AddHash(ref hash, value ? 1 : 0);
    }

    /// <summary>重复检测键：完整局面哈希（已包含当前行动方）</summary>
    public static long RepetitionKey(Gamestate state)
    {
        return StateHash(state);
    }

    // ================================================================
    //  UCB 选择 & 最优行动
    // ================================================================

    /// <summary>PUCT 选择子节点。Scheme A 下 LotteryAction 使用更高的 C。</summary>
    private MctsNode BestPuctChild(MctsNode node)
    {
        MctsNode best = null;
        double bestVal = double.NegativeInfinity;

        foreach (MctsNode child in node.children)
        {
            if (child.pruned) continue;

            double c = (child.action is LotteryAction)
                ? explorationConstant * lotteryCMultiplier
                : explorationConstant;
            double v = child.PuctValue(c, node.visitCount);
            if (v > bestVal)
            {
                bestVal = v;
                best = child;
            }
        }

        return best;
    }

    private GameAction BestChild(MctsNode root)
    {
        MctsNode best = null;
        int bestVisits = -1;

        foreach (MctsNode child in root.children)
        {
            if (child.visitCount > bestVisits)
            {
                bestVisits = child.visitCount;
                best = child;
            }
        }

        return best?.action;
    }

    // ================================================================
    //  模拟策略：吃子优先（权重 3），其他均等
    // ================================================================

    private static GameAction HeuristicPick(List<GameAction> actions,
        Gamestate state, System.Random rng)
    {
        int totalWeight = 0;
        int[] weights = new int[actions.Count];

        for (int i = 0; i < actions.Count; i++)
        {
            int w = ActionWeight(actions[i], state);
            weights[i] = w;
            totalWeight += w;
        }

        int roll = rng.Next(totalWeight);
        int cumulative = 0;
        for (int i = 0; i < actions.Count; i++)
        {
            cumulative += weights[i];
            if (roll < cumulative)
                return actions[i];
        }

        return actions[actions.Count - 1];
    }

    private static int ActionWeight(GameAction action, Gamestate state)
    {
        if (action is MoveAction move)
        {
            Piece target = state[move.toX, move.toY];
            if (target.type != PieceType.Empty && target.type != PieceType.Wall)
                return 3;
            return 1;
        }
        if (action is SniperAction) return 2;
        if (action is LotteryAction) return 1;
        return 1;
    }
}
