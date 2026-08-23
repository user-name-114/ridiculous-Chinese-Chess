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

    public void ExecuteAction(Gamestate state, GameAction action, System.Random rng)
    {
        if (action is LotteryAction)
            ExecuteLottery(state, rng);
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
        int sims = simsOverride ?? maxSimulations;
        MctsNode root = new MctsNode(null, null);

        for (int i = 0; i < sims; i++)
        {
            Gamestate workState;
            try { workState = state.DeepClone(); }
            catch (Exception ex)
            { throw new Exception($"DeepClone failed at sim {i}: {ex.Message}", ex); }

            MctsNode leaf;
            try { leaf = Select(root, workState, rng); }
            catch (Exception ex)
            { throw new Exception($"Select failed at sim {i}: {ex.Message}", ex); }

            double result;
            float[] priors = null;
            try
            {
                if (IsTerminal(workState))
                    result = Evaluate(workState);
                else if (neural != null)
                {
                    var (p, v) = neural.Predict(workState);
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
            { throw new Exception($"Simulate failed at sim {i}: {ex.Message}", ex); }

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
    //  普通节点: Expand 尝试添加子节点 → UCB 选择 → Execute 推进状态
    //  ChanceNode（抽奖）: HandleChanceNode 随机采样 outcome，
    //  创建 outcome 子节点（PW 限制），不提前返回，继续循环深入。
    //  这样可以探索"抽奖后下一步该怎么走"，而非直接 rollout。
    // ================================================================

    private MctsNode Select(MctsNode node, Gamestate state, System.Random rng)
    {
        while (!IsTerminal(state))
        {
            // ── ChanceNode: 随机采样 outcome，创建子节点，继续深入 ──
            if (node.IsChanceNode)
            {
                MctsNode result = HandleChanceNode(node, state, rng);
                if (result == node)
                    return node;          // PW 限制达到，叶节点
                node = result;            // outcome 子节点，继续循环
                continue;
            }

            // ── 未展开（叶节点）→ 返回，交给 RunMctsSingle 展开 + 评估 ──
            if (node.children.Count == 0)
                return node;

            // ── PUCT 选择最优子节点 ──
            node = BestPuctChild(node);

            // ChanceNode 不在此 Execute — HandleChanceNode 会处理，避免双重执行
            if (node.IsChanceNode)
                continue;

            // 普通节点：校验合法性
            if (!IsActionValid(node.action, state))
                return node;

            ExecuteAction(state, node.action, rng);
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

    private MctsNode HandleChanceNode(MctsNode node, Gamestate state, System.Random rng)
    {
        int outcome = rng.Next(1, 41);
        node.sampledOutcome = outcome;

        ExecuteLotteryOutcome(state, outcome, rng);

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

        foreach (LotteryChoice choice in choices)
        {
            Gamestate candidate = state.DeepClone();
            LotteryResolver.ResolveChoice(candidate, outcome, choice);
            GameAction.EndTurn(candidate);

            double opponentValue;
            if (IsTerminal(candidate))
                opponentValue = -Evaluate(candidate);
            else
            {
                var prediction = neural.Predict(candidate);
                opponentValue = candidate.currentTeam == 1 ? prediction.value : -prediction.value;
            }

            if (opponentValue < bestOpponentValue)
            {
                bestOpponentValue = opponentValue;
                bestChoice = choice;
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
    //  零和博弈：在决策节点间翻转 value（对手视角）。
    //  关键：ChanceNode（随机事件）和 outcome 子节点（无 action）
    //  不翻转符号——它们不是对手的决策。
    // ================================================================

    private void Backpropagate(MctsNode node, double result)
    {
        while (node != null)
        {
            node.visitCount++;
            node.totalValue += result;

            // 仅在决策节点翻转（有 action 且非 ChanceNode）
            if (node.action != null && !node.IsChanceNode)
                result = -result;

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

    /// <summary>从最后行动者视角评估: +1 胜 / -1 负 / 0 其他</summary>
    public static double Evaluate(Gamestate state)
    {
        int lastActor = -state.currentTeam;
        bool lastActorKing = HasKing(state, lastActor);
        bool nextKing = HasKing(state, state.currentTeam);

        if (!nextKing && lastActorKing) return +1;
        if (!lastActorKing && nextKing) return -1;
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

    /// <summary>局面哈希（仅棋子位置+类型+阵营），用于重复检测</summary>
    public static long StateHash(Gamestate state)
    {
        long hash = 17;
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty) continue;
                long v = ((long)(int)p.type * 1000000)
                       + ((long)(p.thisTeam + 1) * 100000)
                       + ((long)x * 1000)
                       + (long)y;
                hash = hash * 31 + v;
            }
        return hash;
    }

    /// <summary>重复检测键：棋子布局哈希 + 当前行动方（区分「红走此布局」和「黑走此布局」）</summary>
    public static long RepetitionKey(Gamestate state)
    {
        return StateHash(state) ^ ((long)state.currentTeam << 62);
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
