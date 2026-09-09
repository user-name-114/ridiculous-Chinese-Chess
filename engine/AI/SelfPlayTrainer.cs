using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

// ====================================================================
// 自对弈数据收集器。
//
// 让两个 MCTS AI 互相对弈，记录每一步：
//   - 棋盘特征（StateEncoder.Encode，22×14×11）
//   - 墓地向量（StateEncoder.EncodeGraveyard，18 维）
//   - 稀疏策略分布（MCTS visitCount 归一化，动作索引 + 概率）
//   - 胜负值（+1 红胜 / -1 黑胜，固定红方视角）
//
// 输出：二进制 .bin 文件（每局一个），供 Python train.py 读取。
//
// 并行：外层并行多局（Parallel.For），内层每局 MCTS 用多线程。
//   parallelGames × mctsThreads ≈ CPU 线程数 时利用率最高。
//   例：32 线程 CPU → parallelGames=4, mctsThreads=8。
// ====================================================================

public static class SelfPlayTrainer
{
    // 训练标签中的动作索引上限：移动 23716 + 狙击 616 + 抽奖 1 个标量
    private static readonly int LotteryScalarIndex =
        StateEncoder.MoveActionSize + StateEncoder.SniperActionSize; // 24332

    /// <summary>并行自对弈 numGames 局，数据写到 dataDir。
    /// progressFile：每完成一局写入 "done/total" 供操作面板读取进度。
    /// pauseFlag：存在时暂停（每局开始前检查），供操作面板暂停/继续。</summary>
    public static void Run(int numGames, int numSims, int mctsThreads,
        int parallelGames, string dataDir, string progressFile, string pauseFlag,
        string onnxPath = null, double dirichletAlpha = 0.3,
        double dirichletEpsilon = 0.25, double temperature = 1.0,
        int tempThreshold = 15, double cpuct = 1.2, int maxMoves = 400,
        int neuralBatchSize = 32, int neuralBatchTimeoutMs = 2,
        double evalMaterialWeight = 0.15,
        double virtualLossValue = 0.5, int minGameSamples = 10,
        bool lotteryNnEval = false, bool prepareLottery = false)
    {
        Directory.CreateDirectory(dataDir);

        bool hasOnnxPath = !string.IsNullOrEmpty(onnxPath) && File.Exists(onnxPath);
        if (!string.IsNullOrEmpty(onnxPath) && !hasOnnxPath)
            Console.WriteLine($"[警告] 网络文件不存在，退回纯 MCTS: {onnxPath}");

        // ── 创建一个共享的 NeuralMcts 实例 ──
        // 所有对局共享同一个 ONNX Session + 流水线批量推理。
        NeuralMcts sharedNeural = null;
        if (hasOnnxPath)
        {
            sharedNeural = new NeuralMcts(onnxPath);
            sharedNeural.StartBatchService(neuralBatchSize, neuralBatchTimeoutMs);
            Console.WriteLine($"已加载网络指导自对弈: {onnxPath}");
        }

        // 树内并行（virtual loss）：每局 mctsThreads 个 worker 共享同一棵树，
        // 神经网络模式下 NN 请求密度 ×mctsThreads，批量推理攒得更快。
        int effectiveParallel = parallelGames;
        int effectiveThreads = mctsThreads;

        int totalThreads = effectiveParallel * effectiveThreads;
        System.Threading.ThreadPool.SetMinThreads(totalThreads + 4, totalThreads + 4);

        Console.WriteLine($"自对弈 {numGames} 局 | 每步 {numSims} sims | " +
                          $"{effectiveParallel} 局并行 × {effectiveThreads} MCTS 线程 = {totalThreads} 线程" +
                          (hasOnnxPath
                              ? $" | 树内并行(virtual loss) + 共享 Session 批量推理(batch={neuralBatchSize})"
                              : " | 纯 MCTS"));

        int completed = 0;

        Parallel.For(0, numGames,
            new ParallelOptions { MaxDegreeOfParallelism = effectiveParallel },
            gameIdx =>
            {
                while (File.Exists(pauseFlag))
                    System.Threading.Thread.Sleep(300);

                Console.WriteLine($"开始第 {gameIdx + 1}/{numGames} 局");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                int moves = RunSingleGame(gameIdx, numSims, effectiveThreads, dataDir, sharedNeural,
                    dirichletAlpha, dirichletEpsilon, temperature, tempThreshold, cpuct, maxMoves,
                    pauseFlag, evalMaterialWeight,
            virtualLossValue, minGameSamples, lotteryNnEval, prepareLottery);
                sw.Stop();
                Console.WriteLine($"第 {gameIdx + 1}/{numGames} 局完成，用时 {sw.Elapsed.TotalSeconds:F1} 秒，步数 {moves}");

                int done = System.Threading.Interlocked.Increment(ref completed);
                if (progressFile != null)
                {
                    try { File.WriteAllText(progressFile, $"{done}/{numGames}"); }
                    catch { /* 忽略写进度失败 */ }
                }
            });

        // 所有对局完成后，停止批量推理服务并释放共享 Session
        sharedNeural?.Dispose();

        if (progressFile != null)
            try { File.WriteAllText(progressFile, $"{numGames}/{numGames}"); } catch { }

        Console.WriteLine($"自对弈完成，数据已写入 {dataDir}");
    }

    private static int RunSingleGame(int gameIdx, int numSims, int mctsThreads,
        string dataDir, NeuralMcts neural, double dirichletAlpha,
        double dirichletEpsilon, double temperature, int tempThreshold, double cpuct,
        int maxMoves, string pauseFlag, double evalMaterialWeight,
        double virtualLossValue, int minGameSamples, bool lotteryNnEval,
        bool prepareLottery)
    {
        var red = new AIPlayer(numSims, C: cpuct, seed: gameIdx * 2 + 1,
            aiTeam: 1, threadCount: mctsThreads, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon,
            evalMaterialWeight: evalMaterialWeight,
            virtualLossValue: virtualLossValue,
            lotteryNnEval: lotteryNnEval);
        var black = new AIPlayer(numSims, C: cpuct, seed: gameIdx * 2 + 2,
            aiTeam: -1, threadCount: mctsThreads, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon,
            evalMaterialWeight: evalMaterialWeight,
            virtualLossValue: virtualLossValue,
            lotteryNnEval: lotteryNnEval);

        var state = new Gamestate();
        state.prepareModeOn = false;
        var rng = new Random(gameIdx);

        // ── 2026-09-09（用户要求）：强制开局抽奖（prepareLottery=true 时）──
        // 与对战模式 MatchRunner 相同口径：双方交替各抽 5 次奖（共 10 轮）。
        // 期间 prepareModeOn=true（EndTurn 不恢复狙击冷却/冻结/墙）。
        // 抽奖不计入对局步数（在 move 循环外）、不产生训练样本（无 AI 决策点）、
        // 不计入抽奖数统计、不进重复检测（tracker 在其后创建）。
        if (prepareLottery)
        {
            state.prepareModeOn = true;
            for (int round = 0; round < 10; round++)
            {
                state.currentTeam = (round % 2 == 0) ? 1 : -1;
                int outcome = rng.Next(1, 41);
                LotteryResolver.Resolve(state, outcome, rng);
                state.prepareLotteryCount++;
                if (state.prepareLotteryCount >= 10)
                    state.prepareModeOn = false;
            }
            state.prepareModeOn = false;
            state.currentTeam = 1; // 准备结束，红方先手
            Console.WriteLine($"[Game {gameIdx}] 开局准备：双方各抽 5 次奖（不计入步数/样本/抽奖数）");
        }

        var boards = new List<float[]>();
        var graves = new List<float[]>();
        var policies = new List<(int[] indices, float[] probs)>();
        var legalPerSample = new List<int[]>(); // 2026-09-08 掩码 v2：每决策点的合法动作索引

        int winner = 0;
        string endReason = "撞步数上限（和棋）"; // 循环自然结束时即撞上限
        // 2026-09-08（用户要求）：超短对局剔除阈值（样本数 < 10 即删）
        var repetitionTracker = new RepetitionTracker(state);

        int move;
        for (move = 0; move < maxMoves; move++)
        {
            while (File.Exists(pauseFlag))
                System.Threading.Thread.Sleep(300);

            if (MctsEngine.IsTerminal(state))
            {
                winner = -state.currentTeam; // 当前方被将死，对方胜
                endReason = "吃将";
                Console.WriteLine($"[Game {gameIdx}] 终局=吃将 胜方={(winner == 1 ? "红" : "黑")} 步数={move}");
                break;
            }

            int curTeam = state.currentTeam;
            AIPlayer ai = (curTeam == 1) ? red : black;

            // 编码当前状态
            boards.Add(StateEncoder.Encode(state));
            graves.Add(StateEncoder.EncodeGraveyard(state));

            // 2026-09-08（掩码 v2）：记录当前状态合法动作索引，供训练侧
            // masked softmax 使用。与 GetFilteredActions 同源（自对弈配置
            // allowLottery=true、prepareModeOn=false → 逐字等价），
            // 已由 diag_mask 工具在 817 个决策点实证 target ⊆ legal。
            var legalIdx = new List<int>();
            foreach (var a in ActionGenerator.GetAllActions(state, state.currentTeam))
            {
                int li = EncodeActionForTraining(a);
                if (li >= 0) legalIdx.Add(li);
            }
            legalPerSample.Add(legalIdx.ToArray());

            // 获取 MCTS 动作分布（传入真实对局历史，让 MCTS 感知重复局面）
            var dist = ai.GetActionDistribution(state, repetitionTracker);

            // 稀疏化：只保留概率 > 0 的动作，编码成索引
            var indices = new List<int>();
            var probs = new List<float>();
            foreach (var (action, prob) in dist)
            {
                if (prob <= 1e-6) continue;
                int idx = EncodeActionForTraining(action);
                if (idx < 0) continue;
                indices.Add(idx);
                probs.Add((float)prob);
            }
            policies.Add((indices.ToArray(), probs.ToArray()));

            // 温度采样走子：前 tempThreshold 步用温度 τ 随机，之后贪心
            double temp = (move < tempThreshold) ? temperature : 0.01;
            GameAction best = SampleActionByTemperature(dist, temp, rng);
            if (best == null)
            {
                endReason = "无可用动作";
                Console.WriteLine($"[Game {gameIdx}] 终局=无可用动作 胜方=红 步数={move}");
                break;
            }
            ai.ExecuteAction(state, best);

            // 对局级重复检测：同一局面出现第 3 次，判"刚走的一方"负（抽奖豁免）
            bool isLottery = best is LotteryAction;
            if (repetitionTracker.AddState(state, isLottery))
            {
                winner = state.currentTeam; // 刚走的一方判负，对手胜
                endReason = "重复判负";
                Console.WriteLine($"[Game {gameIdx}] 终局=重复判负 胜方={(winner == 1 ? "红" : "黑")} 步数={move} 走法={best.GetDescription()} isLottery={isLottery}");
                // 打印 MCTS 输出的分布，看是否包含会导致重复的走法
                foreach (var (a, p) in dist)
                    Console.WriteLine($"  动作={a.GetDescription()} 概率={p:F4}");
                winner = state.currentTeam; // 刚走的一方判负，对手胜
                break;
            }
        }

        if (winner == 0)
        {
            // 2026-09-07（第五轮批评 P0-2）：撞步数上限记和棋（value=0），
            // 不再写死红胜。这是训练协议选择（游戏本体无步数上限），
            // 消除方向性标签偏置；终局原因日志同步记录。
            Console.WriteLine($"[Game {gameIdx}] 终局=撞步数上限 和棋 步数={move}");
        }

        WriteData(dataDir, gameIdx, boards, graves, policies, legalPerSample, winner);

        // 2026-09-08（用户要求，方案 A）：剔除超短对局——三五步就结束的对局
        // 没有学习价值，只会污染训练。样本数低于阈值（minGameSamples）的 .bin
        // 立即删除并记录日志；删除不打断自对弈（该局槽位正常释放，继续下一局）。
        if (boards.Count < minGameSamples)
        {
            string badPath = Path.Combine(dataDir, $"game_{gameIdx:D4}.bin");
            if (File.Exists(badPath))
            {
                File.Delete(badPath);
                Console.WriteLine($"[Game {gameIdx}] 终局样本 {boards.Count} < {minGameSamples}，"
                                  + $"对局过短无学习价值，已剔除 .bin（原因：{endReason}）");
            }
        }

        return move;
    }

    /// <summary>把 GameAction 映射到训练标签的动作索引（0~24332）</summary>
    private static int EncodeActionForTraining(GameAction action)
    {
        if (action is MoveAction m)
            return ActionEncoder.EncodeMove(m.fromX, m.fromY, m.toX, m.toY);
        if (action is SniperAction s)
            return ActionEncoder.EncodeSniper(s.fromX, s.fromY, s.dx, s.dy);
        if (action is LotteryAction)
            return LotteryScalarIndex; // 抽奖作为 1 个标量
        return -1;
    }

    /// <summary>按温度 τ 采样走子：p_i^(1/τ) 加权；τ→0 时贪心选概率最大</summary>
    private static GameAction SampleActionByTemperature(
        List<(GameAction action, double probability)> dist, double temp, Random rng)
    {
        if (dist.Count == 0) return null;
        if (temp <= 0.01)
        {
            GameAction best = null;
            double bestP = -1;
            foreach (var (a, p) in dist)
                if (p > bestP) { bestP = p; best = a; }
            return best;
        }
        double total = 0;
        double[] w = new double[dist.Count];
        for (int i = 0; i < dist.Count; i++)
        {
            w[i] = Math.Pow(dist[i].probability, 1.0 / temp);
            total += w[i];
        }
        if (total <= 0) return dist[0].action;
        double roll = rng.NextDouble() * total;
        double cum = 0;
        for (int i = 0; i < dist.Count; i++)
        {
            cum += w[i];
            if (roll < cum) return dist[i].action;
        }
        return dist[dist.Count - 1].action;
    }

    // ================================================================
    //  二进制写入（格式与 train.py 的 load_data 对齐）
    //
    //  [int32] num_samples
    //  [int32] board_feature_size (3388)
    //  [int32] graveyard_size (18)
    //  [int32] version（2；v1 无此字段——v1 首样本 board 首浮点恒 0.0，
    //          其 int 位模式为 0，可安全探测）
    //  每个样本（v2）：
    //    [float32 × 3388] board
    //    [float32 × 18]   graveyard
    //    [int32] nLegal + [int32 × nLegal] legalIdx（合法动作索引，掩码用）
    //    [int32] nTarget + [int32 × nTarget] targetPos（目标在 legalIdx 中的位置）
    //    [float32 × nTarget] probs
    //    [float32] value（红方视角，0=和棋）
    //  目标位置化：训练侧 masked softmax 直接按位置 gather，
    //  与旧格式稀疏 idx/prob 同构；target ⊆ legal 已由
    //  diag_mask 在 817 决策点实证。
    // ================================================================
    private static void WriteData(string dataDir, int gameIdx,
        List<float[]> boards, List<float[]> graves,
        List<(int[] indices, float[] probs)> policies,
        List<int[]> legalPerSample, int winner)
    {
        float value = winner == 1 ? 1f : (winner == -1 ? -1f : 0f); // 红方视角（0=和棋）

        string path = Path.Combine(dataDir, $"game_{gameIdx:D4}.bin");
        using (var fs = new FileStream(path, FileMode.Create))
        using (var w = new BinaryWriter(fs))
        {
            w.Write(boards.Count);
            w.Write(StateEncoder.FeatureSize);
            w.Write(StateEncoder.GraveyardSize);
            w.Write(2); // 格式版本：v2 = 含合法动作掩码

            for (int i = 0; i < boards.Count; i++)
            {
                foreach (float v in boards[i]) w.Write(v);
                foreach (float v in graves[i]) w.Write(v);

                // 合法动作索引（掩码集合，与 GetFilteredActions 同源）
                int[] legal = legalPerSample[i];
                w.Write(legal.Length);
                foreach (int idx in legal) w.Write(idx);

                // 目标对齐：动作索引 → 合法列表内位置（防御：未命中跳过）
                var src = policies[i];
                var posList = new List<int>(src.indices.Length);
                var probList = new List<float>(src.indices.Length);
                for (int k = 0; k < src.indices.Length; k++)
                {
                    int pos = Array.IndexOf(legal, src.indices[k]);
                    if (pos >= 0) { posList.Add(pos); probList.Add(src.probs[k]); }
                }
                w.Write(posList.Count);
                foreach (int pos in posList) w.Write(pos);
                foreach (float p in probList) w.Write(p);

                w.Write(value);
            }
        }
    }
}