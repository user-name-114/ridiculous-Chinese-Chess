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
        int neuralBatchSize = 32, int neuralBatchTimeoutMs = 2)
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

        // ── 扁平化并行（神经网络模式）──
        // 消除嵌套 Parallel.For：parallelGames 个外层线程阻塞等待内层完成，纯浪费。
        // 展开为 parallelGames×mctsThreads 个扁平并行对局，每局 1 线程 MCTS。
        int effectiveParallel = parallelGames;
        int effectiveThreads = mctsThreads;
        if (hasOnnxPath)
        {
            effectiveThreads = 1;
            effectiveParallel = parallelGames * mctsThreads;
            Console.WriteLine($"[神经网络模式] 扁平化: {effectiveParallel} 局 × 1 线程" +
                              $"（原 {parallelGames} × {mctsThreads}，消除 {parallelGames} 个阻塞外层线程）");
        }

        int totalThreads = effectiveParallel * effectiveThreads;
        System.Threading.ThreadPool.SetMinThreads(totalThreads + 4, totalThreads + 4);

        Console.WriteLine($"自对弈 {numGames} 局 | 每步 {numSims} sims | " +
                          $"{effectiveParallel} 局并行 × {effectiveThreads} MCTS 线程 = {totalThreads} 线程" +
                          (hasOnnxPath
                              ? $" | 共享 1 Session + 流水线批量推理(batch={neuralBatchSize})"
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
                    pauseFlag);
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
        int maxMoves, string pauseFlag)
    {
        var red = new AIPlayer(numSims, C: cpuct, seed: gameIdx * 2 + 1,
            aiTeam: 1, threadCount: mctsThreads, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon);
        var black = new AIPlayer(numSims, C: cpuct, seed: gameIdx * 2 + 2,
            aiTeam: -1, threadCount: mctsThreads, neural: neural,
            dirichletAlpha: dirichletAlpha, dirichletEpsilon: dirichletEpsilon);

        var state = new Gamestate();
        state.prepareModeOn = false;
        var rng = new Random(gameIdx);

        var boards = new List<float[]>();
        var graves = new List<float[]>();
        var policies = new List<(int[] indices, float[] probs)>();

        int winner = 0;
        var repetitionTracker = new RepetitionTracker(state);

        int move;
        for (move = 0; move < maxMoves; move++)
        {
            while (File.Exists(pauseFlag))
                System.Threading.Thread.Sleep(300);

            if (MctsEngine.IsTerminal(state))
            {
                winner = -state.currentTeam; // 当前方被将死，对方胜
                break;
            }

            int curTeam = state.currentTeam;
            AIPlayer ai = (curTeam == 1) ? red : black;

            // 编码当前状态
            boards.Add(StateEncoder.Encode(state));
            graves.Add(StateEncoder.EncodeGraveyard(state));

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
            if (best == null) break;
            ai.ExecuteAction(state, best);

            // 对局级重复检测：同一局面出现第 3 次，判"刚走的一方"负（抽奖豁免）
            bool isLottery = best is LotteryAction;
            if (repetitionTracker.AddState(state, isLottery))
            {
                winner = state.currentTeam; // 刚走的一方判负，对手胜
                break;
            }
        }

        if (winner == 0) winner = 1; // 达最大步数，按红方胜处理（罕见）

        WriteData(dataDir, gameIdx, boards, graves, policies, winner);
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
    //  每个样本：
    //    [float32 × 3388] board
    //    [float32 × 18]   graveyard
    //    [int32] num_actions
    //    [int32 × num_actions] indices
    //    [float32 × num_actions] probs
    //    [float32] value
    // ================================================================
    private static void WriteData(string dataDir, int gameIdx,
        List<float[]> boards, List<float[]> graves,
        List<(int[] indices, float[] probs)> policies, int winner)
    {
        float value = winner == 1 ? 1f : -1f; // 红方视角

        string path = Path.Combine(dataDir, $"game_{gameIdx:D4}.bin");
        using (var fs = new FileStream(path, FileMode.Create))
        using (var w = new BinaryWriter(fs))
        {
            w.Write(boards.Count);
            w.Write(StateEncoder.FeatureSize);
            w.Write(StateEncoder.GraveyardSize);

            for (int i = 0; i < boards.Count; i++)
            {
                foreach (float v in boards[i]) w.Write(v);
                foreach (float v in graves[i]) w.Write(v);

                int na = policies[i].indices.Length;
                w.Write(na);
                foreach (int idx in policies[i].indices) w.Write(idx);
                foreach (float p in policies[i].probs) w.Write(p);

                w.Write(value);
            }
        }
    }
}
