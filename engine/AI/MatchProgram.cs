using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

// ====================================================================
// 对战评测程序：加载 onnx 网络，让「网络 AI」与「纯 MCTS AI」对弈。
//
// 每局双方轮流执红（消除先后手偏差），多局并行，统计：
//   - 网络胜/负/和、得分率、Elo 分差（相对纯 MCTS）
//   - 每局步数、双方各自抽奖次数
// 输出：
//   - 控制台实时日志（供操作面板读取）
//   - 对局记录.csv（每局一行）
//   - 汇总.txt（胜率 + Elo + 抽奖统计）
// ====================================================================
public static class MatchProgram
{
    public static void Run(int numGames, string net1, string net2, string mcts2spec,
        string outputDir, int numSims, int mctsThreads, int maxMoves, double cpuct,
        int parallelGames, string progressFile, bool prepareMode = false,
        double evalMaterialWeight = 0.15, double virtualLossValue = 0.5,
        int lotteryEvalLimit = 16,
        string pauseFlag = null)
    {
        Directory.CreateDirectory(outputDir);
        Console.OutputEncoding = Encoding.UTF8;

        bool hasNet2 = !string.IsNullOrEmpty(net2);
        bool netNet = net1 != null && hasNet2;
        bool mctsMcts = net1 == null && !hasNet2;

        string name1 = net1 != null ? Path.GetFileNameWithoutExtension(net1) : "MCTS1";
        string name2 = net2 != null ? Path.GetFileNameWithoutExtension(net2) : "MCTS2";
        if (!netNet && !mctsMcts) name2 = "纯MCTS";

        double c2 = cpuct; int sims2 = numSims; int depth2 = 200;
        double w2 = evalMaterialWeight; double vl2 = virtualLossValue; double mult2 = 1.0;
        if (!string.IsNullOrEmpty(mcts2spec))
            foreach (var kv in mcts2spec.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                var kv2 = kv.Split('=', 2);
                if (kv2.Length != 2) continue;
                var k = kv2[0].Trim().ToLower(); var vs = kv2[1].Trim();
                if (k == "c" && double.TryParse(vs, out var tc)) c2 = tc;
                else if (k == "sims" && int.TryParse(vs, out var ts)) sims2 = ts;
                else if (k == "depth" && int.TryParse(vs, out var td)) depth2 = td;
                else if (k == "w" && double.TryParse(vs, out var tw)) w2 = tw;
                else if (k == "vl" && double.TryParse(vs, out var tvl)) vl2 = tvl;
                else if (k == "mult" && double.TryParse(vs, out var tm)) mult2 = tm;
            }

        string modeDesc = netNet ? "网络1 vs 网络2" : (net1 != null ? "网络 vs 纯MCTS" : "MCTS vs MCTS");
        Console.WriteLine($"对战评测：{modeDesc} × {numGames} 局");
        Console.WriteLine($"玩家1: {name1} | 玩家2: {name2}");
        Console.WriteLine($"每步模拟: {numSims} | 并行: {parallelGames} 局 | 最大步数: {maxMoves}");
        if (mcts2spec.Length > 0) Console.WriteLine($"玩家2覆盖: {mcts2spec}");

        NeuralMcts neuralA = net1 != null ? new NeuralMcts(net1) : null;
        NeuralMcts neuralB = net2 != null ? new NeuralMcts(net2) : null;

        var results = new MatchResult[numGames];
        var aLot = new int[numGames]; var bLot = new int[numGames];
        int completed = 0;

        Parallel.For(0, numGames,
            new ParallelOptions { MaxDegreeOfParallelism = parallelGames },
            i =>
            {
                var state = new Gamestate();
                state.prepareModeOn = false;
                bool aIsRed = (i % 2 == 0);

                var pa = new AIPlayer(numSims, C: cpuct, seed: i * 2 + 1,
                    aiTeam: aIsRed ? 1 : -1, threadCount: mctsThreads, neural: neuralA,
                    evalMaterialWeight: evalMaterialWeight, virtualLossValue: virtualLossValue);
                AIPlayer pb;
                if (hasNet2)
                    pb = new AIPlayer(numSims, C: cpuct, seed: i * 2 + 2,
                        aiTeam: aIsRed ? -1 : 1, threadCount: mctsThreads, neural: neuralB,
                        evalMaterialWeight: evalMaterialWeight, virtualLossValue: virtualLossValue);
                else
                    pb = new AIPlayer(sims2, C: c2, seed: i * 2 + 2,
                        aiTeam: aIsRed ? -1 : 1, threadCount: mctsThreads,
                        maxRolloutDepth: depth2, evalMaterialWeight: w2,
                        virtualLossValue: vl2, lotteryCMultiplier: mult2);

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.Run(state, aIsRed ? pa : pb,
                    aIsRed ? pb : pa, maxMoves: maxMoves, recordSteps: false,
                    prepareMode: prepareMode,
                    repetitionLoser: 0, pauseFlag: pauseFlag);
                sw.Stop();

                results[i] = result;
                aLot[i] = aIsRed ? result.redLotteryCount : result.blackLotteryCount;
                bLot[i] = aIsRed ? result.blackLotteryCount : result.redLotteryCount;

                string outcome = result.winner == 0 ? "和"
                    : (result.winner == (aIsRed ? 1 : -1) ? "P1胜" : "P1负");
                int done = Interlocked.Increment(ref completed);
                Console.WriteLine($"第 {i + 1}/{numGames} 局(完成序 {done}): {outcome}（{result.totalMoves}步 | {sw.Elapsed.TotalSeconds:F1}s | {name1}抽奖{aLot[i]} vs {name2}抽奖{bLot[i]}）");
                if (progressFile != null)
                    try { File.WriteAllText(progressFile, $"{done}/{numGames}"); } catch { }
            });

        int aWins = 0, aLoss = 0, draws = 0;
        int aLotT = 0, bLotT = 0;
        var csv = new StringBuilder();
        csv.AppendLine($"局号,{name1}执子,结果,步数,{name1}抽奖次数,{name2}抽奖次数,结束原因");
        for (int i = 0; i < numGames; i++)
        {
            bool aIsRed = (i % 2 == 0);
            var r = results[i];
            string outcome;
            if (r.winner == 0) { outcome = "和"; draws++; }
            else if (r.winner == (aIsRed ? 1 : -1)) { outcome = $"{name1}胜"; aWins++; }
            else { outcome = $"{name1}负"; aLoss++; }
            aLotT += aLot[i]; bLotT += bLot[i];
            string endReason = r.endReason switch
            {
                "king_captured" => "将死",
                "重复局面" => "重复判负",
                _ => "判和",
            };
            csv.AppendLine($"{i + 1},{(aIsRed ? "先手" : "后手")},{outcome},{r.totalMoves},{aLot[i]},{bLot[i]},{endReason}");
        }

        double score = (aWins + 0.5 * draws) / numGames;
        double eloDiff = EloDiff(score);

        Console.WriteLine($"=== 汇总 ===");
        Console.WriteLine($"{name1} 胜 {aWins} | 负 {aLoss} | 和 {draws}");
        Console.WriteLine($"得分率: {score:P2}");
        Console.WriteLine($"Elo 分差: {eloDiff:+0;-0;0}（相对 {name2}）");
        Console.WriteLine($"{name1} 总抽奖 {aLotT} 次 | {name2} 总抽奖 {bLotT} 次");

        string folderName = Path.GetFileName(outputDir.TrimEnd('\\', '/'));
        File.WriteAllText(Path.Combine(outputDir, $"{folderName}对局记录.csv"),
            csv.ToString(), new UTF8Encoding(true));
        var summary = new StringBuilder();
        summary.AppendLine("=== 对战评测汇总 ===");
        summary.AppendLine($"玩家1: {name1} | 玩家2: {name2}");
        summary.AppendLine($"对局数: {numGames}");
        summary.AppendLine($"{name1} 胜: {aWins}  负: {aLoss}  和: {draws}");
        summary.AppendLine($"得分率: {score:P2}");
        summary.AppendLine($"Elo 分差: {eloDiff:+0;-0;0}");
        summary.AppendLine($"{name1} 总抽奖: {aLotT} 次");
        summary.AppendLine($"{name2} 总抽奖: {bLotT} 次");
        File.WriteAllText(Path.Combine(outputDir, "汇总.txt"), summary.ToString(),
            new UTF8Encoding(true));
        Console.WriteLine($"结果已保存: {outputDir}");
    }

    /// <summary>把得分率换算成 Elo 分差（相对对手），处理 0%/100% 的边界。</summary>
    private static double EloDiff(double score)
    {
        if (score >= 0.999) return 800;
        if (score <= 0.001) return -800;
        return -400 * Math.Log10(1.0 / score - 1);
    }

}
