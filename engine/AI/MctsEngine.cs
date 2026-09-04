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
    private Queue<(long key, bool isReal)> recent = new Queue<(long key, bool isReal)>();
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
        recent.Enqueue((key, true));
        counts.TryGetValue(key, out int c);
        counts[key] = c + 1;

        while (recent.Count > WindowSize)
        {
            var (oldKey, isReal) = recent.Dequeue();
            if (!isReal) continue;
            counts[oldKey]--;
            if (counts[oldKey] == 0)
                counts.Remove(oldKey);
        }
    }

    /// <summary>抽奖走法：滑动窗口但不增加重复计数</summary>
    private void AddKeyLottery(long key)
    {
        ply++;
        recent.Enqueue((key, false));
        // 不增加 counts

        while (recent.Count > WindowSize)
        {
            var (oldKey, isReal) = recent.Dequeue();
            if (!isReal) continue;
            counts[oldKey]--;
            if (counts[oldKey] == 0)
                counts.Remove(oldKey);
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

    // ── 诊断计数器（静态，跨引擎实例共享；ResetStats 后按次读取差值）──
    internal static long StatRollouts, StatRolloutTerm, StatSweeps, StatSweepCand, StatChanceSel, StatNewPairs;
    internal static long StatPhaseCloneDescend, StatPhaseNN, StatPhaseRoll, StatPhaseExpandBack;
    internal static long StatPhaseExpandPre, StatPhaseExpandLock, StatPhaseWorkerWall;
    // 2026-09-04 起全部计时器改 ElapsedTicks 累计（消除 (long)ms 截断），读取时 /Stopwatch.Frequency*1000
    internal static void ResetStats() { StatRollouts = StatRolloutTerm = StatSweeps = StatSweepCand = StatChanceSel = StatNewPairs = 0; StatPhaseCloneDescend = StatPhaseNN = StatPhaseRoll = StatPhaseExpandBack = StatPhaseExpandPre = StatPhaseExpandLock = StatPhaseWorkerWall = 0; }

    private double evalMaterialWeight;
    private int lotteryEvalLimit;

    public MctsEngine(int simulations = 1000, double C = 1.2, int maxRolloutDepth = 200,
        bool allowLottery = true, int aiTeam = 0,
        double lotteryCMultiplier = 1.0, bool useLotteryChanceNodes = false,
        int threadCount = 1, NeuralMcts neural = null,
        double dirichletAlpha = 0, double dirichletEpsilon = 0,
        double evalMaterialWeight = 0.15,
        double virtualLossValue = 0.5,
        int lotteryEvalLimit = 16)
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
        this.evalMaterialWeight = evalMaterialWeight;
        this.virtualLossValue = virtualLossValue;
        this.lotteryEvalLimit = lotteryEvalLimit;
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

    /// <summary>运行 MCTS（带真实对局历史），返回最优行动</summary>
    public GameAction FindBestAction(Gamestate state, System.Random rng, RepetitionTracker history)
    {
        MctsNode root = RunMcts(state, rng, history);
        return BestChild(root);
    }

    /// <summary>运行 MCTS，返回根节点各行动的概率分布（visitCount 归一化）</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(
        Gamestate state, System.Random rng)
    {
        MctsNode root = RunMcts(state, rng, null);
        return BuildDistribution(root);
    }

    /// <summary>运行 MCTS（带真实对局历史），返回概率分布</summary>
    public List<(GameAction action, double probability)> GetActionDistribution(
        Gamestate state, System.Random rng, RepetitionTracker history)
    {
        MctsNode root = RunMcts(state, rng, history);
        if (root.visitCount == 0)
        {
            // 所有模拟都被剪枝，回退到无历史搜索
            return GetActionDistribution(state, rng);
        }
        return BuildDistribution(root);
    }

    /// <summary>从根节点构建概率分布，过滤掉 pruned 子节点</summary>
    private List<(GameAction action, double probability)> BuildDistribution(MctsNode root)
    {
        var dist = new List<(GameAction action, double probability)>();
        double total = 0;
        foreach (MctsNode child in root.children)
        {
            if (child.pruned) continue;
            total += child.visitCount;
        }
        if (total == 0) return dist;
        foreach (MctsNode child in root.children)
        {
            if (child.pruned) continue;
            dist.Add((child.action, child.visitCount / total));
        }
        return dist;
    }

    /// <summary>
    /// 【诊断专用】返回根节点每个子动作的统计信息（含被 pruned 的），
    /// 用于排查“抽奖从未被选中”这类问题。不改变任何搜索行为。
    /// Q 为执行该动作一方视角的均值。
    /// </summary>
    public List<(GameAction action, string desc, bool isChance,
                 double prior, int visits, double q, bool pruned)> GetRootChildStats(
        Gamestate state, System.Random rng, RepetitionTracker history)
    {
        MctsNode root = RunMcts(state, rng, history);
        var stats = new List<(GameAction, string, bool, double, int, double, bool)>();
        foreach (MctsNode c in root.children)
        {
            double q = c.visitCount > 0 ? c.totalValue / c.visitCount : 0.0;
            stats.Add((c.action, c.action?.GetDescription() ?? "(null)", c.IsChanceNode,
                       c.prior, c.visitCount, q, c.pruned));
        }
        return stats;
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
        return RunMcts(state, rng, null);
    }

    /// <summary>根并行化 MCTS，支持传入真实对局历史。</summary>
    private MctsNode RunMcts(Gamestate state, System.Random rng, RepetitionTracker history)
    {
        if (threadCount <= 1)
            return RunMctsSingle(state, rng, null, history);
        return RunMctsTreeParallel(state, rng, history);
    }

    private readonly object _treeLock = new object();
    private double virtualLossValue;

    private MctsNode RunMctsTreeParallel(Gamestate state, System.Random rng,
        RepetitionTracker history)
    {
        var root = new MctsNode(null, null);
        int threads = Math.Min(threadCount, maxSimulations);

        var seeds = new int[threads];
        lock (rng)
        {
            for (int t = 0; t < threads; t++) seeds[t] = rng.Next();
        }

        int per = maxSimulations / threads;
        int rem = maxSimulations % threads;
        // 专用线程而非 ThreadPool：避免嵌套并行下的线程注入延迟，
        // 确保 K 个 worker 真正同时进入 NN 等待（提高批量密度与 GPU 占用）。
        // virtual loss 补偿机制保证统计口径与串行一致，不影响棋力。
        var handles = new System.Threading.Thread[threads];
        for (int w = 0; w < threads; w++)
        {
            int wid = w;
            handles[w] = new System.Threading.Thread(() =>
            {
                int mySims = per + (wid < rem ? 1 : 0);
                var wr = new System.Random(seeds[wid]);
                for (int s = 0; s < mySims; s++)
                    WorkerSim(root, state, history, wr);
            });
            handles[w].Start();
        }
        foreach (var h in handles) h.Join();
        return root;
    }

    /// <summary>
    /// 单个 worker 执行一次完整模拟。虚拟损失在选择经过每条边时立即扣减，
    /// BackpropAggregate 回传时补偿并累加真实结果 —— 统计口径与串行一致。
    /// </summary>
    private void WorkerSim(MctsNode root, Gamestate rootState,
        RepetitionTracker history, System.Random wr)
    {
        var simRepeatSkip = new HashSet<MctsNode>();   // worker 本地临时跳过集，不写共享节点（2026-09-04 修复）
        var __wall = System.Diagnostics.Stopwatch.StartNew();   // 整个 WorkerSim（单次模拟）总墙钟
        try
        {
            var __ph = System.Diagnostics.Stopwatch.StartNew();
var ws = rootState.DeepClone();              // 线程私有工作副本
            var trk = history?.Clone() ?? new RepetitionTracker();
            var node = root;
            var path = new List<MctsNode>(64);           // 本 sim 经过的已扣 VL 边
            MctsNode leafFound = null;

            while (true)
            {
                if (IsTerminal(ws)) break;               // 树中途撞上终局

                // ── ChanceNode：采样 outcome，锁内确保唯一创建 ──
                                if (node.IsChanceNode)
                {
                    int outcome = wr.Next(1, 41);
                    ((LotteryAction)node.action).lastOutcome = outcome;

                    // ── 方案①（2026-09-04 修复）：候选枚举与评估移出临界区 ──
                    // 第一段锁只查字典；未命中时在锁外完成 O(n²) 候选枚举与
                    // SelectLotteryChoiceByValue 评估（仅依赖 worker 私有 ws，无共享状态），
                    // 再进第二段短锁二次检查并建节点。并发未命中时可能重复计算，
                    // 先建者胜、后者复用已建节点（仅浪费计算，无正确性影响）。
                    MctsNode oc = null;
                    bool miss = false;
                    LotteryChoice preFc = null;
                    lock (_treeLock)
                    {
                        if (node.outcomeChildren == null)
                            node.outcomeChildren = new Dictionary<int, MctsNode>();
                        miss = !node.outcomeChildren.TryGetValue(outcome, out oc);
                    }
                    if (miss)
                    {
                        var chs = LotteryResolver.GetChoices(ws, outcome);
                        if (chs.Count == 0)
                            preFc = null;
                        else if (neural == null)
                            preFc = chs[wr.Next(chs.Count)];
                        else
                        {
                            // 候选可能成百上千（双生成炮马类目标对）：全量评估会造成评估风暴，超限等距抽样
                            chs = SampleChoices(chs, lotteryEvalLimit, wr);
                            preFc = SelectLotteryChoiceByValue(ws, outcome, chs);
                        }
                    }
                    lock (_treeLock)
                    {
                        if (!node.outcomeChildren.TryGetValue(outcome, out oc))
                        {
                            oc = new MctsNode(null, node) { fixedChoice = preFc };
                            node.outcomeChildren[outcome] = oc;
                        }
                        oc.visitCount++;
                        oc.totalValue -= virtualLossValue;
                        path.Add(oc);
                    }

                    if (oc.fixedChoice == null)
                        LotteryResolver.ResolveChoice(ws, outcome, null);
                    else
                        LotteryResolver.ResolveChoice(ws, outcome, oc.fixedChoice);
                    GameAction.EndTurn(ws);
                    trk.AddState(ws, isLottery: true);
                    node = oc;
                    continue;
                }
                

                // ── 普通 / 叶判定与选路（统计变更段加锁）──
                bool reachedLeaf = false;
                bool exhausted = false;
                MctsNode nextNode = null;
                MctsNode pendingChance = null;
                GameAction pendingEdge = null;

                lock (_treeLock)
                {
                    if (node.children.Count == 0)
                    {
                        leafFound = node;                 // 出锁后评估+展开
                        reachedLeaf = true;
                    }
                    else
                    {
                        while (true)
                        {
                            var child = BestPuctChild(node, wr, simRepeatSkip);
                            if (child == null) { exhausted = true; break; }
                            if (child.IsChanceNode)
                            {
                                StatChanceSel++;
                                child.visitCount++;
                                child.totalValue -= virtualLossValue;
                                path.Add(child);
                                pendingChance = child;    // 不在此执行
                                node = child;
                                break;
                            }
                            if (!IsActionValid(child.action, ws))
                            {
                                child.pruned = true;      // 非法是永久的
                                continue;
                            }
                            var ts = ws.DeepClone();
                            ExecuteAction(ts, child.action, wr);
                            if (trk.WouldRepeat(ts))
                            {
                                                                simRepeatSkip.Add(child); // 仅本 sim 本地跳过（2026-09-04 修复）
                                continue;
                            }
                            child.visitCount++;
                            child.totalValue -= virtualLossValue;
                            path.Add(child);
                            nextNode = child;
                            pendingEdge = child.action;
                            node = child;
                            break;
                        }
                    }
                }

                simRepeatSkip.Clear();

                if (reachedLeaf || exhausted) break;

                if (pendingChance != null) { continue; }  // 执行交由下一轮 Chance 分支

                ExecuteAction(ws, pendingEdge, wr);       // 普通边确认执行（锁外）
                trk.AddState(ws, false);
                _ = nextNode;                              // node 已在锁内前移
            }

            System.Threading.Interlocked.Add(ref StatPhaseCloneDescend, __ph.ElapsedTicks);
            __ph.Restart();
            // ── 叶评估（全部在锁外）──
            double result;
            float[] priors = null;
            if (IsTerminal(ws))
                result = Evaluate(ws);
            else if (neural != null)
            {
                var __nn = System.Diagnostics.Stopwatch.StartNew();
                var (pol, vNet) = neural.PredictBlocking(ws);
                __nn.Stop();
                System.Threading.Interlocked.Add(ref StatPhaseNN, __nn.ElapsedTicks);
                priors = pol;
                result = ws.currentTeam == -1 ? -vNet : vNet;
            }
            else
            {
                var __rl = System.Diagnostics.Stopwatch.StartNew();
                result = Simulate(ws, wr);
                __rl.Stop();
                System.Threading.Interlocked.Add(ref StatPhaseRoll, __rl.ElapsedTicks);
            }

            // 展开（若刚才到达的是未展开叶；存在竞态时幂等跳过）
            var __ex = System.Diagnostics.Stopwatch.StartNew();
            Gamestate expandState = ws.DeepClone();   // 修复(位置已核正)：快照=叶子原局面（Select 之后）
            long __exPre = __ex.ElapsedTicks;
            System.Threading.Interlocked.Add(ref StatPhaseExpandPre, __exPre);
            if (!IsTerminal(expandState) && leafFound != null && leafFound.children.Count == 0)
            {
                lock (_treeLock)
                {
                    if (leafFound.children.Count == 0 && !IsTerminal(expandState))
                    {
                        ExpandAll(leafFound, expandState, priors);
                        if (leafFound == root && dirichletAlpha > 0 && dirichletEpsilon > 0)
                            AddDirichletNoise(root, wr);
                    }
                }
                System.Threading.Interlocked.Add(ref StatPhaseExpandLock, __ex.ElapsedTicks - __exPre);
            }

            var __eb = System.Diagnostics.Stopwatch.StartNew();
            BackpropAggregate(path, leafFound, result, root);
            __eb.Stop();
            System.Threading.Interlocked.Add(ref StatPhaseExpandBack, __eb.ElapsedTicks);
        }
        finally
        {
            System.Threading.Interlocked.Add(ref StatPhaseWorkerWall, __wall.ElapsedTicks);
        }
    }

    /// <summary>
    /// 回传（并行版，2026-09-04 重写）：①补偿路径上所有虚拟损失（含叶自身）；
    /// ②按串行一致的翻转规则自叶向上逐动作节点翻转并累加一次真实结果——
    /// 叶节点 visitCount 已在 descend 选边时计入，本方法仅给根节点 visitCount+1
    /// （保持 PUCT parentVisits 口径与串行版一致）。
    /// </summary>
        private void BackpropAggregate(List<MctsNode> path, MctsNode leaf,
        double result, MctsNode root)
    {
        lock (_treeLock)
        {
            // 修复（2026-09-03，外部代码审查确认）：旧实现把 leaf 单独处理了一遍——
            //   leaf.visitCount 二次自增、totalValue 被 -result/+result 抵消、
            //   整条链多翻转一次导致所有祖先符号反转。改为与串行 Backpropagate 对齐：
            //   descend 时已计入 visitCount/-VL，此处只补偿 VL 并沿 path 逐动作翻转累加一次。
            foreach (var n in path)
                n.totalValue += virtualLossValue;         // 补偿选择期的 -VL（含叶节点自身）

            double r = result;
            for (int i = path.Count - 1; i >= 0; i--)     // 逆序：末位即叶节点
            {
                var n = path[i];
                if (n.action != null) r = -r;
                n.totalValue += r;
            }

            root.visitCount++;
            root.totalValue += r;                          // root.action == null，不翻转
        }
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
            var prunedThisSelect = new List<MctsNode>();

            MctsNode leaf;
            try { leaf = Select(root, workState, rng, simTracker, prunedThisSelect); }
            catch (Exception ex)
            { throw new Exception($"Select failed at sim {validSims}: {ex.Message}", ex); }
            finally
            {
                // 异常安全清除临时剪枝标记（重复剪枝只在单次模拟内有效）；
                // 与“合法性失败”的永久标记区分开，防止临时标记泄漏成永久剪枝
                foreach (var n in prunedThisSelect)
                    n.pruned = false;
            }

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
            Gamestate expandState = workState.DeepClone();   // 修复(位置已核正)：快照=叶子原局面（Select 之后）
            if (!IsTerminal(expandState) && leaf.children.Count == 0 && !leaf.IsChanceNode)
            {
                ExpandAll(leaf, expandState, priors);
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
        RepetitionTracker repetitionTracker, List<MctsNode> prunedThisSelect)
    {
        while (!IsTerminal(state))
        {
            // ── ChanceNode: 随机采样 outcome，创建子节点，继续深入 ──
            if (node.IsChanceNode)
            {
                // HandleChanceNode 采样 outcome、执行（固化 choice 重放）并返回 outcome 子节点。
                // 抽奖的执行只发生在这里；Select 的“确认执行”段必须跳过 ChanceNode，
                // 否则会叠加执行两次抽奖效果（历史重大 bug）
                node = HandleChanceNode(node, state, rng, repetitionTracker);
                continue;
            }

            // ── 未展开（叶节点）→ 返回，交给 RunMctsSingle 展开 + 评估 ──
            if (node.children.Count == 0)
                return node;

            // ── PUCT 选择最优子节点 ──
            // 重复检测：实时检查 WouldRepeat，不永久标记 pruned。
            // 同一节点在不同模拟路径中重复条件不同，永久标记会导致错误剪枝。
            MctsNode child = null;
            while (true)
            {
                child = BestPuctChild(node, rng);
                if (child == null)
                    return null; // 所有走法都不可用

                // 抽奖子节点不检查重复，直接使用
                if (child.IsChanceNode)
                    break;

                // 普通节点：校验合法性
                if (!IsActionValid(child.action, state))
                {
                    child.pruned = true; // 合法性失败是永久的（棋子已移动/冻结等）
                    continue;
                }

                // 检查重复：试走，看是否构成第 3 次重复
                Gamestate testState = state.DeepClone();
                ExecuteAction(testState, child.action, rng);
                if (repetitionTracker.WouldRepeat(testState))
                {
                    // 本次模拟中此走法会导致重复，跳过（不永久标记）
                    // 用临时标记让 BestPuctChild 跳过，本次模拟结束后清除
                    child.pruned = true;
                    prunedThisSelect.Add(child);
                    continue;
                }

                break; // 找到合法且不重复的子节点
            }

            // 确认执行：推进真实 state 和 tracker。
            // 注意：ChanceNode（抽奖）子节点不在此执行——
            // 它的执行（含重复豁免计数）由 HandleChanceNode 统一负责，
            // 这里若执行会造成双重抽奖效果叠加
            if (!child.IsChanceNode)
            {
                ExecuteAction(state, child.action, rng);
                repetitionTracker.AddState(state);
            }

            node = child;
        }

        return node; // 终局
    }

    // ================================================================
    //  HandleChanceNode — Scheme B 抽奖随机节点
    //
    //  均匀随机采样 outcome (1~40)，执行效果，推进 state。
    //  outcome 子节点按需懒创建（每个 ChanceNode 至多 40 个），
    //  首建时固化本 outcome 实际执行的 LotteryChoice，重放时原样复现，
    //  保证同一 outcome 子节点永远对应同一个具体局面序列。
    //  返回 outcome 子节点（继续深入）。
    // ================================================================

    private MctsNode HandleChanceNode(MctsNode node, Gamestate state, System.Random rng,
        RepetitionTracker repetitionTracker)
    {
        int outcome = rng.Next(1, 41);
        node.sampledOutcome = outcome;
        ((LotteryAction)node.action).lastOutcome = outcome;

        // 查找已有 outcome 子节点
        if (node.outcomeChildren == null)
            node.outcomeChildren = new Dictionary<int, MctsNode>();

        if (!node.outcomeChildren.TryGetValue(outcome, out MctsNode oc))
        {
            // 新建 outcome 子节点：确定并固化本次实际执行的 LotteryChoice。
            // 纯 MCTS 下 choice 选择带随机性；此后每次经过该子节点必须原样重放，
            // 否则同一子节点对应不同具体局面，破坏子树状态一致性（历史 bug 根源）
            List<LotteryChoice> choices = LotteryResolver.GetChoices(state, outcome);
            // 候选可能成百上千（双生成枚举目标对）：全量评估会造成评估风暴，超限等距抽样
            // ── 已知问题（2026-09-04 外部审查核实，按约定注释搁置）──
            // 问题：纯 MCTS（neural == null）且走串行路径（threadCount <= 1）时，本方法
            // 先随机无放回抽样到 lotteryEvalLimit 再随机取一个，候选分布是"抽样子集内均匀"；
            // 并行版 WorkerSim 的对应分支则是全量随机（不抽样）。两版行为不等价，
            // 且此处对纯随机选择而言抽样步骤是纯浪费。
            // 搁置原因：① 当前训练与对局均走神经网络模式，纯 MCTS + 串行路径基本不用；
            // ② 修复需先统一三处口径（本方法 / WorkerSim / ExecuteLotteryOutcome），
            //    单改此处会制造新的不一致；③ 收益低、回归风险高。
            choices = SampleChoices(choices, lotteryEvalLimit, rng);
            LotteryChoice fixedChoice;
            if (choices.Count == 0)
                fixedChoice = null; // 无可选目标，交给 Resolver 的自动路径
            else
                fixedChoice = neural == null
                    ? choices[rng.Next(choices.Count)]
                    : SelectLotteryChoiceByValue(state, outcome, choices);

            oc = new MctsNode(null, node) { fixedChoice = fixedChoice }; // outcome 子节点无 action
            node.outcomeChildren[outcome] = oc;
        }

        // 执行效果（重放固化的 choice 或自动路径）+ 回合切换。
        // 注意此处不再每次到达重新随机选 choice —— 修复纯 MCTS 下
        // 同一 outcome 子节点对应不同具体局面的问题
        LotteryResolver.ResolveChoice(state, outcome, oc.fixedChoice);
        GameAction.EndTurn(state);

        // 抽奖豁免重复检测——不剪枝、不判负
        repetitionTracker.AddState(state, isLottery: true);

        return oc;
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

        // 修复（2026-09-04，批评确认）：与搜索树内候选逻辑对齐——神经网络模式下
        // 同样等距抽样到 lotteryEvalLimit，保证树内固化的 choice 与真实执行
        // 产生自同一候选集；同时消除真实对局数百次 DeepClone+评估的阻塞。
        if (neural != null)
            choices = SampleChoices(choices, lotteryEvalLimit, rng);

        LotteryChoice selected = neural == null
            ? choices[rng.Next(choices.Count)]
            : SelectLotteryChoiceByValue(state, outcome, choices);
        LotteryResolver.ResolveChoice(state, outcome, selected);
    }

    /// <summary>
    /// 候选效果选择：用 F1 子力差启发式静态评估（选使对手局面最差的效果）。
    /// 【性能关键】不再调用神经网络——此函数在搜索内部高频触发，
    /// 旧版直连 session.Run 与批量流水线争抢 GPU，是 44s/步 的主要阻塞源。
    /// F1 材料差对"生成/升级类"效果是足够好的短期代理；长期价值仍由
    /// 主搜索的 outcome 子树统计学习。
    /// </summary>
    private LotteryChoice SelectLotteryChoiceByValue(Gamestate state, int outcome,
        List<LotteryChoice> choices)
    {
        StatSweeps++;
        StatSweepCand += choices.Count;
        LotteryChoice bestChoice = choices[0];
        double bestOpponentValue = double.PositiveInfinity;

        for (int i = 0; i < choices.Count; i++)
        {
            var candidate = state.DeepClone();
            LotteryResolver.ResolveChoice(candidate, outcome, choices[i]);
            GameAction.EndTurn(candidate);

            double opponentValue;
            if (IsTerminal(candidate))
                opponentValue = Evaluate(candidate);
            else
                opponentValue = EvalMaterialHeuristic(candidate);

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
        StatRollouts++;
        for (int depth = 0; depth < maxRolloutDepth; depth++)
        {
            if (IsTerminal(state))
            {
                StatRolloutTerm++;
                return Evaluate(state);
            }

            // 轻量随机走子（不创建 GameAction 对象，避免 GC 压力）
            if (!LightRandomMove(state, rng))
                break;
        }

        // F1：未分胜负时不再返回死平的 0，改用子力差启发式给搜索梯度
        return EvalMaterialHeuristic(state);
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

    // 【甲·战术化】随机选一个合法目标（最多尝试 8 次）：
    // 命中吃子立即优先执行（含吃王 → 真实终局 ±1 信号）；
    // 安静着法先记住，未命中吃子时才兜底执行。
            int quietTx = -1, quietTy = -1;
            for (int attempt2 = 0; attempt2 < 8; attempt2++)
            {
                int tx = rng.Next(xMin, xMax + 1);
                int ty = rng.Next(yMin, yMax + 1);
                if (!piece.IsLegalMove(tx, ty, state))
                    continue;

                Piece target = state[tx, ty];
                if (target.type != PieceType.Empty)
                {
                    // 吃子：立即执行，不与安静着法同权竞争
                    piece.Move(tx, ty, state);
                    if (target.isDead)
                        state.AddToGraveyard(target);
                    GameAction.EndTurn(state);
                    return true;
                }
                if (quietTx < 0)
                {
                    quietTx = tx;
                    quietTy = ty;
                }
            }
            if (quietTx >= 0)
            {
                piece.Move(quietTx, quietTy, state);
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
            // 边界检查：action 坐标可能在搜索树持久化期间因领域展开/收缩而失效
            if (!state.IsValidPosition(m.fromX, m.fromY) || !state.IsValidPosition(m.toX, m.toY))
                return false;
            Piece piece = state[m.fromX, m.fromY];
            if (piece.type == PieceType.Empty || piece.thisTeam != action.team)
                return false;
            if (piece.frozenTurns > 0) return false;
            return piece.IsLegalMove(m.toX, m.toY, state);
        }
        if (action is SniperAction s)
        {
            if (!state.IsValidPosition(s.fromX, s.fromY))
                return false;
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

    /// <summary>
    /// 【F1·非终局启发式】按子力价值差给出连续信号，修复纯MCTS“全零价值塌缩”。
    /// 仅影响 rollout 走满/中断时的兑底估值；终局仍由 Evaluate 精确判定。
    /// 输出压在 [-w,+w]，真实终局 ±1 永远占主导；权重可由面板参数调节。
    /// </summary>
    private double EvalMaterialHeuristic(Gamestate state)
    {
        if (evalMaterialWeight <= 0) return 0;
        double mine = 0, theirs = 0;
        int cur = state.currentTeam;
        for (int x = state.leftBound; x <= state.rightBound; x++)
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty || p.isDead) continue;
                double v = PieceStrengthPoints(p);
                if (v <= 0) continue;
                if (p.thisTeam == cur) mine += v; else theirs += v;
            }
        return evalMaterialWeight * Math.Tanh((mine - theirs) / 8.0);
    }

    /// <summary>
    /// 棋子强度点数表（F1）——按玩家实战体验排序标定，名称↔等级对照
    /// 以 RuleEngine.cs 的 GetDescription 为准。
    /// 将/墙/空格不计入：将死由 Evaluate 判定，墙是临时物。
    /// </summary>
    private static double PieceStrengthPoints(Piece p)
    {
        switch (p.type)
        {
            case PieceType.Rook:
                return p.upgradeLevel >= 1 ? 16.0 : 8.2;          // 赛车 / 车
            case PieceType.Cannon:
                switch (p.upgradeLevel)
                {
                    case 1: return 11.0;                           // 炮车
                    case 2: return 10.5;                           // 迫击炮
                    case 3: return 15.5;                           // 迫击炮车（炮车+迫击炮叠加）
                    default: return 6.2;                           // 炮
                }
            case PieceType.Knight:
                return p.upgradeLevel >= 1 ? 6.0 : 4.2;            // 连环马 / 马
            case PieceType.Bishop:
                switch (p.upgradeLevel)
                {
                    case 1: return 2.8;                            // 巨象
                    case 2: return 3.1;                            // 小飞象
                    case 3: return 3.5;                            // 巨飞象
                    default: return 2.5;                           // 象
                }
            case PieceType.Guard:
                return p.upgradeLevel >= 1 ? 3.4 : 2.1;            // 武士 / 士
            case PieceType.Pawn:
                switch (p.upgradeLevel)
                {
                    case 1: return 8.8;                            // 狙击兵
                    case 2: return 3.9;                            // 自爆兵
                    case 3: return 9.5;                            // 狙击自爆兵
                    default: return 1.6;                           // 兵
                }
            default:
                return 0.0;
        }
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

    /// <summary>PUCT 选择子节点。Scheme A 下 LotteryAction 使用更高的 C。
    /// F2：完全平手时随机破序，消除动作枚举顺序导致的结构性饿死
    /// （抽奖是枚举末位，在零值世界里会被系统性跳过）。</summary>
    /// <summary>
    /// 无放回均匀抽样（Fisher-Yates 部分洗牌取前 k 个）。
    /// 修复（2026-09-04，批评确认）：双生成类候选由 for i, for j>i 嵌套循环生成，
    /// 索引→(i,j) 映射不均匀（小 i 的 pair 密集、大 i 稀疏），原按索引等距取样
    /// 会系统性偏向列表前段；改为随机无放回抽样后所有候选等概率进入评估子集。
    /// </summary>
    private static List<LotteryChoice> SampleChoices(List<LotteryChoice> chs, int k, System.Random rng)
    {
        if (chs.Count <= k) return chs;
        var pool = new List<LotteryChoice>(chs);
        for (int i = 0; i < k; i++)
        {
            int jj = rng.Next(i, pool.Count);
            (pool[i], pool[jj]) = (pool[jj], pool[i]);
        }
        return pool.GetRange(0, k);
    }

    private MctsNode BestPuctChild(MctsNode node, System.Random rng, HashSet<MctsNode> skip = null)
    {
        double bestVal = double.NegativeInfinity;
        var tied = new List<MctsNode>();

        foreach (MctsNode child in node.children)
        {
            if (child.pruned || (skip != null && skip.Contains(child))) continue;

            double c = (child.action is LotteryAction)
                ? explorationConstant * lotteryCMultiplier
                : explorationConstant;
            double v = child.PuctValue(c, node.visitCount);
            if (v > bestVal) bestVal = v;
        }

        foreach (MctsNode child in node.children)
        {
            if (child.pruned || (skip != null && skip.Contains(child))) continue;

            double c = (child.action is LotteryAction)
                ? explorationConstant * lotteryCMultiplier
                : explorationConstant;
            double v = child.PuctValue(c, node.visitCount);
            if (v >= bestVal - 1e-9)
                tied.Add(child);
        }

        return tied.Count == 0 ? null : tied[rng.Next(tied.Count)];
    }

    private GameAction BestChild(MctsNode root)
    {
        MctsNode best = null;
        int bestVisits = -1;

        foreach (MctsNode child in root.children)
        {
            if (child.pruned) continue;
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
