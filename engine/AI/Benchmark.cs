using System;
using System.Collections.Generic;
using System.IO;

// ====================================================================
// 棋力基准测试。纯 C#，可在 Unity 或独立控制台中运行。
//作用：不参与实际游戏流程，而是用于离线评估 AI 棋力的调试工具。
//RunAiVsRandom：让 AI 与纯随机走子对手进行多局对弈，统计胜率、平均耗时等。
//RunAiVsAi：让两个 AI 互相对弈，用于观察相同配置下的棋力稳定性。
//RunCFocusedRetest：针对几个候选 C 值做重测（20局/值），除胜率外还统计平均步数和抽奖次数，用于评估 C 值对 AI 风格的影响。
//RunCFinalTest：对候选 C 值做终测（25局/值），输出两份 CSV 文件（汇总和逐局详情），供 Python 绘图分析。
//RunCParameterSweep：对一组 C 值进行全面扫参（10局/值），输出 CSV 供分析最佳探索常数。
//辅助 WriteCsv：简单的 CSV 写入工具。
// ====================================================================
public static class Benchmark
{
    public static void RunAiVsRandom(int aiSimulations = 200, int matchCount = 10, double C = 0.5)
    {
        var ai = new AIPlayer(simulations: aiSimulations, C: C, seed: 42);

        int aiWins = 0, aiLosses = 0, draws = 0;
        var totalTimes = TimeSpan.Zero;

        Console.WriteLine($"=== AI({aiSimulations} sims, C={C}) vs Random × {matchCount} ===");

        for (int i = 0; i < matchCount; i++)
        {
            var state = new Gamestate();
            bool aiIsRed = (i % 2 == 0);
            int aiTeam = aiIsRed ? 1 : -1;

            ai.ResetRandom(i);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = MatchRunner.RunVsRandom(state, ai, aiTeam, maxMoves: 300);
            sw.Stop();
            totalTimes += sw.Elapsed;

            string outcome;
            if (result.winner == aiTeam) { aiWins++; outcome = "AI 胜"; }
            else if (result.winner == -aiTeam) { aiLosses++; outcome = "AI 负"; }
            else { draws++; outcome = "和"; }

            Console.WriteLine($"  局{i + 1}: {outcome} ({result.totalMoves}步, {sw.Elapsed.TotalSeconds:F1}s)");
        }

        Console.WriteLine($"  → 胜率: {aiWins}/{matchCount} ({100.0 * aiWins / matchCount:F0}%)  负: {aiLosses}  和: {draws}  平均: {totalTimes.TotalSeconds / matchCount:F1}s/局");
    }

    public static void RunAiVsAi(int simulations = 200, int matchCount = 4)
    {
        int redWins = 0, blackWins = 0, draws = 0;
        var totalTimes = TimeSpan.Zero;

        Console.WriteLine($"=== AI({simulations} sims) vs AI({simulations} sims) × {matchCount} ===");

        for (int i = 0; i < matchCount; i++)
        {
            var red = new AIPlayer(simulations, C: 0.5, seed: i * 2);
            var black = new AIPlayer(simulations, C: 0.5, seed: i * 2 + 1);
            var state = new Gamestate();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = MatchRunner.Run(state, red, black, maxMoves: 300, recordSteps: false);
            sw.Stop();
            totalTimes += sw.Elapsed;

            if (result.winner == 1) { redWins++; Console.WriteLine($"  局{i + 1}: 红胜 ({result.totalMoves}步)"); }
            else if (result.winner == -1) { blackWins++; Console.WriteLine($"  局{i + 1}: 黑胜 ({result.totalMoves}步)"); }
            else { draws++; Console.WriteLine($"  局{i + 1}: 和  ({result.totalMoves}步)"); }

            Console.WriteLine($"    耗时 {sw.Elapsed.TotalSeconds:F1}s");
        }

        Console.WriteLine($"---");
        Console.WriteLine($"  红胜: {redWins}  黑胜: {blackWins}  和: {draws}");
        Console.WriteLine($"  平均耗时: {totalTimes.TotalSeconds / matchCount:F1}s/局");
    }

    // ================================================================
    //  C 值聚焦重测：20 局/值，含步数和抽奖次数
    // ================================================================
    public static void RunCFocusedRetest(int aiSimulations = 100)
    {
        double[] cValues = { 0.5, 1.4, 2.0 };
        int matchCount = 20;

        Console.WriteLine($"=== C 值聚焦重测 ({aiSimulations} sims × {matchCount} 局/C) ===\n");

        foreach (double c in cValues)
        {
            int wins = 0, losses = 0, draws = 0;
            double totalTime = 0;
            int totalMoves = 0, totalLottery = 0;
            int minMoves = int.MaxValue, maxMoves = 0;
            int minLottery = int.MaxValue, maxLottery = 0;

            for (int i = 0; i < matchCount; i++)
            {
                var ai = new AIPlayer(simulations: aiSimulations, C: c, seed: i);
                var state = new Gamestate();
                bool aiIsRed = (i % 2 == 0);
                int aiTeam = aiIsRed ? 1 : -1;

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.RunVsRandom(state, ai, aiTeam, maxMoves: 300);
                sw.Stop();

                totalTime += sw.Elapsed.TotalSeconds;
                totalMoves += result.totalMoves;
                totalLottery += result.aiLotteryCount;
                if (result.totalMoves < minMoves) minMoves = result.totalMoves;
                if (result.totalMoves > maxMoves) maxMoves = result.totalMoves;
                if (result.aiLotteryCount < minLottery) minLottery = result.aiLotteryCount;
                if (result.aiLotteryCount > maxLottery) maxLottery = result.aiLotteryCount;

                if (result.winner == aiTeam) wins++;
                else if (result.winner == -aiTeam) losses++;
                else draws++;
            }

            double avgMoves = (double)totalMoves / matchCount;
            double avgLottery = (double)totalLottery / matchCount;

            Console.WriteLine($"  C={c:F1}");
            Console.WriteLine($"    胜率: {wins}/{matchCount} ({100.0 * wins / matchCount:F0}%)  负: {losses}  和: {draws}");
            Console.WriteLine($"    平均步数: {avgMoves:F0} ({minMoves}~{maxMoves})");
            Console.WriteLine($"    平均抽奖次数: {avgLottery:F1} ({minLottery}~{maxLottery})");
            Console.WriteLine($"    平均耗时: {totalTime / matchCount:F1}s/局");
            Console.WriteLine();
        }
    }

    // ================================================================
    //  C 值终测：25 局/C，输出详细 CSV 供 Python 绘图
    // ================================================================
    public static void RunCFinalTest(int aiSimulations = 100, int matchCount = 25)
    {
        double[] cValues = { 0.5, 1.4, 2.0 };
        var summaryRows = new List<string[]>();
        var detailRows = new List<string[]>();

        summaryRows.Add(new[] { "C", "WinRate(%)", "Wins", "Losses", "Draws",
            "AvgMoves", "MinMoves", "MaxMoves",
            "AvgLottery", "MinLottery", "MaxLottery",
            "AvgLotteryRatio(%)", "AvgTime(s)" });
        detailRows.Add(new[] { "C", "Game", "Win", "TotalMoves", "AILotteryCount", "LotteryRatio(%)", "Time(s)" });

        Console.WriteLine($"=== C 值终测 ({aiSimulations} sims × {matchCount} 局/C) ===\n");

        foreach (double c in cValues)
        {
            int wins = 0, losses = 0, draws = 0;
            double totalTime = 0;
            int totalMoves = 0, totalLottery = 0;
            double totalLotteryRatio = 0;
            int minMoves = int.MaxValue, maxMoves = 0;
            int minLottery = int.MaxValue, maxLottery = 0;

            for (int i = 0; i < matchCount; i++)
            {
                var ai = new AIPlayer(simulations: aiSimulations, C: c, seed: i);
                var state = new Gamestate();
                bool aiIsRed = (i % 2 == 0);
                int aiTeam = aiIsRed ? 1 : -1;

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.RunVsRandom(state, ai, aiTeam, maxMoves: 300);
                sw.Stop();

                int win = (result.winner == aiTeam) ? 1 : 0;
                double ratio = result.totalMoves > 0
                    ? 100.0 * result.aiLotteryCount / result.totalMoves : 0;

                totalTime += sw.Elapsed.TotalSeconds;
                totalMoves += result.totalMoves;
                totalLottery += result.aiLotteryCount;
                totalLotteryRatio += ratio;
                if (result.totalMoves < minMoves) minMoves = result.totalMoves;
                if (result.totalMoves > maxMoves) maxMoves = result.totalMoves;
                if (result.aiLotteryCount < minLottery) minLottery = result.aiLotteryCount;
                if (result.aiLotteryCount > maxLottery) maxLottery = result.aiLotteryCount;

                if (result.winner == aiTeam) wins++;
                else if (result.winner == -aiTeam) losses++;
                else draws++;

                detailRows.Add(new[] {
                    c.ToString("F1"), i.ToString(), win.ToString(),
                    result.totalMoves.ToString(), result.aiLotteryCount.ToString(),
                    ratio.ToString("F1"), sw.Elapsed.TotalSeconds.ToString("F1")
                });

                string status = win == 1 ? "胜" : (result.winner == 0 ? "和" : "负");
                Console.WriteLine($"  C={c:F1} 局{i + 1}: {status}  {result.totalMoves}步  抽{result.aiLotteryCount}次  {sw.Elapsed.TotalSeconds:F1}s");
            }

            double avgMoves = (double)totalMoves / matchCount;
            double avgLottery = (double)totalLottery / matchCount;
            double avgRatio = totalLotteryRatio / matchCount;

            Console.WriteLine($"  ── C={c:F1} 汇总 ──");
            Console.WriteLine($"    胜: {wins}/{matchCount} ({100.0 * wins / matchCount:F0}%)  负: {losses}  和: {draws}");
            Console.WriteLine($"    步数: {avgMoves:F0} ({minMoves}~{maxMoves})");
            Console.WriteLine($"    抽奖: {avgLottery:F1}次/局 ({minLottery}~{maxLottery})  占比: {avgRatio:F1}%");
            Console.WriteLine($"    耗时: {totalTime / matchCount:F1}s/局");
            Console.WriteLine();

            summaryRows.Add(new[] {
                c.ToString("F1"),
                (100.0*wins/matchCount).ToString("F0"),
                wins.ToString(), losses.ToString(), draws.ToString(),
                avgMoves.ToString("F0"), minMoves.ToString(), maxMoves.ToString(),
                avgLottery.ToString("F1"), minLottery.ToString(), maxLottery.ToString(),
                avgRatio.ToString("F1"), (totalTime/matchCount).ToString("F1")
            });
        }

        // 写 CSV
        string dir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";

        WriteCsv(Path.Combine(dir, "c_final_summary.csv"), summaryRows);
        WriteCsv(Path.Combine(dir, "c_final_detail.csv"), detailRows);
    }

    // ================================================================
    //  C 值精细扫参：自定义 C 列表，输出详细 CSV
    // ================================================================
    public static void RunCFineSweep(double[] cValues, int aiSimulations = 100, int matchCount = 20)
    {
        var detailRows = new List<string[]>();
        detailRows.Add(new[] { "C", "Game", "Win", "TotalMoves", "AILotteryCount", "LotteryRatio(%)", "Time(s)" });

        Console.WriteLine($"=== C Fine Sweep ({aiSimulations} sims × {matchCount} games/C) ===\n");

        foreach (double c in cValues)
        {
            int wins = 0, losses = 0, draws = 0;
            double totalTime = 0;
            int totalMoves = 0;

            for (int i = 0; i < matchCount; i++)
            {
                var ai = new AIPlayer(simulations: aiSimulations, C: c, seed: i);
                var state = new Gamestate();
                bool aiIsRed = (i % 2 == 0);
                int aiTeam = aiIsRed ? 1 : -1;

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.RunVsRandom(state, ai, aiTeam, maxMoves: 300);
                sw.Stop();

                int win = (result.winner == aiTeam) ? 1 : 0;
                double ratio = result.totalMoves > 0
                    ? 100.0 * result.aiLotteryCount / result.totalMoves : 0;

                totalTime += sw.Elapsed.TotalSeconds;
                totalMoves += result.totalMoves;

                if (result.winner == aiTeam) wins++;
                else if (result.winner == -aiTeam) losses++;
                else draws++;

                detailRows.Add(new[] {
                    c.ToString("F1"), i.ToString(), win.ToString(),
                    result.totalMoves.ToString(), result.aiLotteryCount.ToString(),
                    ratio.ToString("F1"), sw.Elapsed.TotalSeconds.ToString("F1")
                });

                string status = win == 1 ? "W" : (result.winner == 0 ? "D" : "L");
                Console.Write($"  C={c:F1}[{i+1}]{status}({result.totalMoves}m) ");
                if ((i + 1) % 10 == 0) Console.WriteLine();
            }

            Console.WriteLine($"\n  => C={c:F1}: {wins}/{matchCount} ({100.0*wins/matchCount:F0}%)  "
                + $"avg {totalMoves/matchCount:F0} moves  {totalTime/matchCount:F1}s/game\n");
        }

        string dir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        WriteCsv(Path.Combine(dir, "c_fine_detail.csv"), detailRows);
    }

    // ================================================================
    //  Scheme A (higher C for lottery) vs Scheme B (chance nodes)
    // ================================================================
    public static void RunSchemeAvsB(int aiSimulations = 100, int matchCount = 30)
    {
        var detailRows = new List<string[]>();
        detailRows.Add(new[] { "Game", "SchemeAWon", "SchemeATeam", "TotalMoves",
            "SchemeALottery", "SchemeBLottery", "Time(s)" });

        Console.WriteLine($"=== Scheme A (C×5) vs Scheme B (Chance Node)  "
            + $"({aiSimulations} sims × {matchCount} games) ===\n");

        int aWins = 0, bWins = 0, draws = 0;
        double totalTime = 0;

        for (int i = 0; i < matchCount; i++)
        {
            int aTeam = (i % 2 == 0) ? 1 : -1;
            int bTeam = -aTeam;

            var aiA = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2,
                allowLottery: true, aiTeam: aTeam,
                lotteryCMultiplier: 5.0, useLotteryChanceNodes: false);

            var aiB = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2 + 1,
                allowLottery: true, aiTeam: bTeam,
                lotteryCMultiplier: 1.0, useLotteryChanceNodes: true);

            AIPlayer red = (aTeam == 1) ? aiA : aiB;
            AIPlayer black = (aTeam == -1) ? aiA : aiB;

            var state = new Gamestate();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = MatchRunner.Run(state, red, black, maxMoves: 300, recordSteps: false);
            sw.Stop();

            totalTime += sw.Elapsed.TotalSeconds;
            bool aWon = result.winner == aTeam;
            int aLottery = (aTeam == 1) ? result.redLotteryCount : result.blackLotteryCount;
            int bLottery = (aTeam == 1) ? result.blackLotteryCount : result.redLotteryCount;

            if (result.winner == aTeam) aWins++;
            else if (result.winner == bTeam) bWins++;
            else draws++;

            string outcome = aWon ? "A-WIN" : (result.winner == bTeam ? "B-WIN" : "DRAW");
            detailRows.Add(new[] {
                i.ToString(), aWon ? "1" : "0", aTeam == 1 ? "R" : "B",
                result.totalMoves.ToString(), aLottery.ToString(),
                bLottery.ToString(), sw.Elapsed.TotalSeconds.ToString("F1")
            });

            Console.WriteLine($"  Game{i+1}: {outcome}  A_lot={aLottery}  B_lot={bLottery}  "
                + $"{result.totalMoves}m  {sw.Elapsed.TotalSeconds:F1}s");
        }

        Console.WriteLine($"\n=== Summary ===");
        Console.WriteLine($"  Scheme A (Cx5):  {aWins}/{matchCount} ({100.0*aWins/matchCount:F0}%)");
        Console.WriteLine($"  Scheme B (Chance): {bWins}/{matchCount} ({100.0*bWins/matchCount:F0}%)");
        Console.WriteLine($"  Draws: {draws}  Avg: {totalTime/matchCount:F1}s/game\n");

        string dir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        WriteCsv(Path.Combine(dir, "scheme_a_vs_b.csv"), detailRows);
    }

    private static void WriteCsv(string path, List<string[]> rows)
    {
        try
        {
            using var w = new StreamWriter(path);
            foreach (var row in rows)
                w.WriteLine(string.Join(",", row));
            Console.WriteLine($"CSV saved: {Path.GetFullPath(path)}");
        }
        catch (Exception ex) { Console.WriteLine($"Write failed {path}: {ex.Message}"); }
    }

    // ================================================================
    //  Scheme B (Chance Node) vs No-Lottery AI
    // ================================================================
    public static void RunSchemeBvsNoLottery(int aiSimulations = 100, int matchCount = 25,
        int threadCount = 8, string csvName = "scheme_b_vs_nolottery.csv",
        bool prepareMode = false)
    {
        var detailRows = new List<string[]>();
        detailRows.Add(new[] { "Game", "LotteryWon", "LotteryTeam", "TotalMoves",
            "LotteryCount", "NoLotteryCount", "Time(s)" });

        Console.WriteLine($"=== Scheme B (Chance) vs No-Lottery AI  "
            + $"({aiSimulations} sims × {matchCount} games) ===\n");

        int lotWins = 0, noLotWins = 0, draws = 0;
        double totalTime = 0;

        for (int i = 0; i < matchCount; i++)
        {
            int lotteryTeam = (i % 2 == 0) ? 1 : -1;
            int noLotteryTeam = -lotteryTeam;

            var lotAi = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2,
                allowLottery: true, aiTeam: lotteryTeam,
                useLotteryChanceNodes: true, threadCount: threadCount);
            var noLotAi = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2 + 1,
                allowLottery: false, aiTeam: noLotteryTeam,
                useLotteryChanceNodes: true, threadCount: threadCount);

            AIPlayer red = (lotteryTeam == 1) ? lotAi : noLotAi;
            AIPlayer black = (lotteryTeam == -1) ? lotAi : noLotAi;

            var state = new Gamestate();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = MatchRunner.Run(state, red, black, maxMoves: 400,
                recordSteps: false, prepareMode: prepareMode);
            sw.Stop();

            totalTime += sw.Elapsed.TotalSeconds;
            bool lotWon = result.winner == lotteryTeam;
            int lotCount = (lotteryTeam == 1) ? result.redLotteryCount : result.blackLotteryCount;
            int noCount = (lotteryTeam == 1) ? result.blackLotteryCount : result.redLotteryCount;

            if (result.winner == lotteryTeam) lotWins++;
            else if (result.winner == noLotteryTeam) noLotWins++;
            else draws++;

            string outcome = lotWon ? "L-WIN" : (result.winner == noLotteryTeam ? "N-WIN" : "DRAW");
            detailRows.Add(new[] {
                i.ToString(), lotWon ? "1" : "0", lotteryTeam == 1 ? "R" : "B",
                result.totalMoves.ToString(), lotCount.ToString(),
                noCount.ToString(), sw.Elapsed.TotalSeconds.ToString("F1")
            });

            Console.WriteLine($"  Game{i+1}: {outcome}  lot={lotCount}  noLot={noCount}  "
                + $"{result.totalMoves}m  {sw.Elapsed.TotalSeconds:F1}s");
        }

        Console.WriteLine($"\n=== Summary ===");
        Console.WriteLine($"  Lottery AI (Scheme B): {lotWins}/{matchCount} ({100.0*lotWins/matchCount:F0}%)");
        Console.WriteLine($"  No-Lottery AI:         {noLotWins}/{matchCount} ({100.0*noLotWins/matchCount:F0}%)");
        Console.WriteLine($"  Draws: {draws}  Avg time: {totalTime/matchCount:F1}s\n");

        string dir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        string outPath = Path.GetFullPath(Path.Combine(dir, "..", "..", "..", "..",
            "Assets", "Scripts", "core", "AI", csvName));
        WriteCsv(outPath, detailRows);
    }

    // ================================================================
    //  Lottery vs No-Lottery AI 对比测试
    // ================================================================
    public static void RunLotteryVsNoLottery(int aiSimulations = 100, int matchCount = 25)
    {
        var detailRows = new List<string[]>();
        detailRows.Add(new[] { "Game", "LotteryWon", "LotteryTeam", "TotalMoves",
            "LotteryCount", "NoLotteryCount", "Time(s)" });

        Console.WriteLine($"=== Lottery AI vs No-Lottery AI ({aiSimulations} sims × {matchCount} games) ===\n");

        int lotteryWins = 0, noLotteryWins = 0, draws = 0;
        double totalTime = 0;
        int totalMoves = 0;

        for (int i = 0; i < matchCount; i++)
        {
            // alternate sides
            int lotteryTeam = (i % 2 == 0) ? 1 : -1;
            int noLotteryTeam = -lotteryTeam;

            var lotteryAi = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2,
                allowLottery: true, aiTeam: lotteryTeam);
            var noLotteryAi = new AIPlayer(simulations: aiSimulations, C: 1.2, seed: i * 2 + 1,
                allowLottery: false, aiTeam: noLotteryTeam);

            AIPlayer red = (lotteryTeam == 1) ? lotteryAi : noLotteryAi;
            AIPlayer black = (lotteryTeam == -1) ? lotteryAi : noLotteryAi;

            var state = new Gamestate();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = MatchRunner.Run(state, red, black, maxMoves: 500, recordSteps: false);
            sw.Stop();

            totalTime += sw.Elapsed.TotalSeconds;
            totalMoves += result.totalMoves;

            bool lotteryWon = result.winner == lotteryTeam;
            int lotteryCount = (lotteryTeam == 1) ? result.redLotteryCount : result.blackLotteryCount;
            int noLotteryCount = (lotteryTeam == 1) ? result.blackLotteryCount : result.redLotteryCount;

            if (result.winner == lotteryTeam) lotteryWins++;
            else if (result.winner == noLotteryTeam) noLotteryWins++;
            else draws++;

            string outcome = result.winner == lotteryTeam ? "L-WIN" :
                             (result.winner == noLotteryTeam ? "N-WIN" : "DRAW");

            detailRows.Add(new[] {
                i.ToString(),
                lotteryWon ? "1" : "0",
                lotteryTeam == 1 ? "R" : "B",
                result.totalMoves.ToString(),
                lotteryCount.ToString(),
                noLotteryCount.ToString(),
                sw.Elapsed.TotalSeconds.ToString("F1")
            });

            Console.WriteLine($"  Game{i+1}: {outcome}  lottery={lotteryCount}  noLottery={noLotteryCount}  "
                + $"{result.totalMoves}m  {sw.Elapsed.TotalSeconds:F1}s");
        }

        Console.WriteLine($"\n=== Summary ===");
        Console.WriteLine($"  Lottery AI wins: {lotteryWins}/{matchCount} ({100.0*lotteryWins/matchCount:F0}%)");
        Console.WriteLine($"  No-Lottery AI wins: {noLotteryWins}/{matchCount} ({100.0*noLotteryWins/matchCount:F0}%)");
        Console.WriteLine($"  Draws: {draws}");
        Console.WriteLine($"  Avg moves: {totalMoves/matchCount:F0}  Avg time: {totalTime/matchCount:F1}s\n");

        string dir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        WriteCsv(Path.Combine(dir, "lottery_vs_nolottery.csv"), detailRows);
    }

    // ================================================================
    //  C 值扫参：AI vs Random，输出 CSV
    // ================================================================
    public static void RunCParameterSweep(int aiSimulations = 100, int matchCount = 10)
    {
        double[] cValues = { 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5, 3.0 };
        var rows = new List<string[]>();
        rows.Add(new[] { "C", "WinRate(%)", "Wins", "Losses", "Draws",
                         "AvgTime(s)", "AvgMoves", "MinMoves", "MaxMoves" });

        Console.WriteLine($"=== C Parameter Sweep ({aiSimulations} sims × {matchCount} matches) ===");

        foreach (double c in cValues)
        {
            int wins = 0, losses = 0, draws = 0;
            double totalTime = 0;
            int totalMoves = 0, minMoves = int.MaxValue, maxMoves = 0;

            for (int i = 0; i < matchCount; i++)
            {
                var ai = new AIPlayer(simulations: aiSimulations, C: c, seed: i);
                var state = new Gamestate();
                bool aiIsRed = (i % 2 == 0);
                int aiTeam = aiIsRed ? 1 : -1;

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = MatchRunner.RunVsRandom(state, ai, aiTeam, maxMoves: 300);
                sw.Stop();

                totalTime += sw.Elapsed.TotalSeconds;
                totalMoves += result.totalMoves;
                if (result.totalMoves < minMoves) minMoves = result.totalMoves;
                if (result.totalMoves > maxMoves) maxMoves = result.totalMoves;

                if (result.winner == aiTeam) wins++;
                else if (result.winner == -aiTeam) losses++;
                else draws++;
            }

            double winRate = 100.0 * wins / matchCount;
            double avgTime = totalTime / matchCount;
            double avgMoves = (double)totalMoves / matchCount;

            Console.WriteLine($"  C={c,4:F1}:  {winRate,5:F0}% ({wins}/{matchCount})  "
                + $"avg {avgTime:F1}s  moves {avgMoves:F0} ({minMoves}~{maxMoves})");

            rows.Add(new[] {
                c.ToString("F1"),
                winRate.ToString("F0"),
                wins.ToString(),
                losses.ToString(),
                draws.ToString(),
                avgTime.ToString("F1"),
                avgMoves.ToString("F0"),
                minMoves.ToString(),
                maxMoves.ToString()
            });
        }

        // 写 CSV
        string csvPath = Path.Combine(
            Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)
            ?? ".",
            "c_parameter_sweep.csv");

        // 同时写到 AI 文件夹，方便找到
        string aiPath = "c_parameter_sweep.csv";

        foreach (string path in new[] { csvPath, aiPath })
        {
            try
            {
                using var writer = new StreamWriter(path);
                foreach (var row in rows)
                    writer.WriteLine(string.Join(",", row));
                Console.WriteLine($"\n结果已写入: {Path.GetFullPath(path)}");
            }
            catch { /* 忽略写入失败 */ }
        }
    }
}
