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
    public static void Run(int numGames, string onnxPath, string outputDir,
        int numSims, int mctsThreads, int maxMoves, double cpuct,
        int parallelGames, string progressFile, bool prepareMode = false)
    {
        Directory.CreateDirectory(outputDir);
        Console.OutputEncoding = Encoding.UTF8;

        Console.WriteLine($"对战评测：onnx vs 纯 MCTS × {numGames} 局");
        Console.WriteLine($"网络: {onnxPath}");
        Console.WriteLine($"每步模拟: {numSims} | 并行: {parallelGames} 局 | 最大步数: {maxMoves}");

        NeuralMcts neural = new NeuralMcts(onnxPath);

        // 每局结果（并行填充）
        var results = new MatchResult[numGames];
        var onnxLotteries = new int[numGames];
        var mctsLotteries = new int[numGames];
        int completed = 0;

        Parallel.For(0, numGames,
            new ParallelOptions { MaxDegreeOfParallelism = parallelGames },
            i =>
            {
                var state = new Gamestate();
                state.prepareModeOn = false;
                bool onnxIsRed = (i % 2 == 0);

                var onnxAI = new AIPlayer(numSims, C: cpuct, seed: i * 2 + 1,
                    aiTeam: onnxIsRed ? 1 : -1, threadCount: mctsThreads, neural: neural);
                var mctsAI = new AIPlayer(numSims, C: cpuct, seed: i * 2 + 2,
                    aiTeam: onnxIsRed ? -1 : 1, threadCount: mctsThreads);

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.Run(state, onnxIsRed ? onnxAI : mctsAI,
                    onnxIsRed ? mctsAI : onnxAI, maxMoves: maxMoves, recordSteps: false,
                    prepareMode: prepareMode,
                    repetitionLoser: onnxIsRed ? 1 : -1); // 重复局面只判 onnx 负，避免靠对手先重复获胜
                sw.Stop();

                results[i] = result;
                onnxLotteries[i] = onnxIsRed ? result.redLotteryCount : result.blackLotteryCount;
                mctsLotteries[i] = onnxIsRed ? result.blackLotteryCount : result.redLotteryCount;

                string outcome = result.winner == 0 ? "和"
                    : (result.winner == (onnxIsRed ? 1 : -1) ? "onnx胜" : "onnx负");

                int done = Interlocked.Increment(ref completed);
                Console.WriteLine($"第 {done}/{numGames} 局: {outcome}（{result.totalMoves}步 | 用时{sw.Elapsed.TotalSeconds:F1}s | onnx抽奖{onnxLotteries[i]} vs MCTS抽奖{mctsLotteries[i]}）");

                if (progressFile != null)
                    try { File.WriteAllText(progressFile, $"{done}/{numGames}"); } catch { }
            });

        // ── 汇总 ──
        int onnxWins = 0, onnxLosses = 0, draws = 0;
        int onnxLotteryTotal = 0, mctsLotteryTotal = 0;
        var csv = new StringBuilder();
        csv.AppendLine("局号,onnx执子,结果,步数,onnx抽奖次数,纯MCTS抽奖次数,结束原因");

        for (int i = 0; i < numGames; i++)
        {
            bool onnxIsRed = (i % 2 == 0);
            var r = results[i];
            string outcome;
            if (r.winner == 0) { outcome = "和"; draws++; }
            else if (r.winner == (onnxIsRed ? 1 : -1)) { outcome = "onnx胜"; onnxWins++; }
            else { outcome = "onnx负"; onnxLosses++; }

            onnxLotteryTotal += onnxLotteries[i];
            mctsLotteryTotal += mctsLotteries[i];
            string endReason = r.endReason switch
            {
                "king_captured" => "将死",
                "重复局面" => "重复判负",
                _ => "判和",
            };
            csv.AppendLine($"{i + 1},{(onnxIsRed ? "红" : "黑")},{outcome},{r.totalMoves},{onnxLotteries[i]},{mctsLotteries[i]},{endReason}");
        }

        double score = (onnxWins + 0.5 * draws) / numGames;
        double eloDiff = EloDiff(score);

        Console.WriteLine($"=== 汇总 ===");
        Console.WriteLine($"胜 {onnxWins} | 负 {onnxLosses} | 和 {draws}");
        Console.WriteLine($"得分率: {score:P2}（胜 + 0.5×和）");
        Console.WriteLine($"Elo 分差: {eloDiff:+0;-0;0}（相对纯 MCTS）");
        Console.WriteLine($"onnx 总抽奖 {onnxLotteryTotal} 次 | 纯 MCTS 总抽奖 {mctsLotteryTotal} 次");

        string folderName = Path.GetFileName(outputDir.TrimEnd('\\', '/'));
        File.WriteAllText(Path.Combine(outputDir, $"{folderName}对局记录.csv"), csv.ToString(), new UTF8Encoding(true));
        var summary = new StringBuilder();
        summary.AppendLine("=== 对战评测汇总 ===");
        summary.AppendLine($"网络文件: {onnxPath}");
        summary.AppendLine($"对局数: {numGames}");
        summary.AppendLine($"胜: {onnxWins}  负: {onnxLosses}  和: {draws}");
        summary.AppendLine($"得分率: {score:P2}（胜 + 0.5×和）");
        summary.AppendLine($"Elo 分差: {eloDiff:+0;-0;0}（相对纯 MCTS，和棋折半）");
        summary.AppendLine($"onnx 总抽奖: {onnxLotteryTotal} 次");
        summary.AppendLine($"纯 MCTS 总抽奖: {mctsLotteryTotal} 次");
        File.WriteAllText(Path.Combine(outputDir, "汇总.txt"), summary.ToString(), new UTF8Encoding(true));

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
