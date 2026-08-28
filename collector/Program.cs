using System;
using System.IO;
using System.Text.Json;

// ====================================================================
// 入口：支持两种模式。
//
// 模式 1（默认）：自对弈数据收集
//   用法: SelfPlayCollector <numGames> <dataDir> <progressFile> <pauseFlag> [onnxPath]
//     numGames:     对局数
//     dataDir:      数据输出目录
//     progressFile: 进度文件（每完成一局写 "done/total"）
//     pauseFlag:    暂停标志文件（存在则暂停）
//     onnxPath:     （可选）上一代网络的 ONNX 路径，指导自对弈（AlphaZero 迭代）
//
// 模式 2：对战评测
//   用法: SelfPlayCollector match <numGames> <onnxPath> <outputDir> [progressFile]
//     numGames:     对局数
//     onnxPath:     要评测的网络 ONNX 路径
//     outputDir:    输出目录（CSV + 汇总）
//     progressFile: 进度文件
//
// 从 ../config.json 读取 num_mcts_sims 等超参数。
// ====================================================================

// 读 config.json（两种模式共用）
var configPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "config.json"));
int numSims = 200;
int parallelGames = 8;
int mctsThreads = 4;
double dirichletAlpha = 0.3;
double dirichletEpsilon = 0.25;
double temperature = 1.0;
int tempThreshold = 15;
double cpuct = 1.2;
int maxMoves = 400;
double evalMaterialWeight = 0.15;
double virtualLossValue = 0.5;
int lotteryEvalLimit = 16;
int neuralBatchSize = 32;
int neuralBatchTimeoutMs = 2; // ms
try
{
    using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
    var mcts = doc.RootElement.GetProperty("mcts");
    numSims = mcts.GetProperty("num_mcts_sims").GetInt32();
    temperature = mcts.GetProperty("temperature").GetDouble();
    tempThreshold = mcts.GetProperty("temp_threshold").GetInt32();
    cpuct = mcts.GetProperty("cpuct").GetDouble();
    if (mcts.TryGetProperty("eval_material_weight", out var emwEl))
        evalMaterialWeight = emwEl.GetDouble();
    if (mcts.TryGetProperty("virtual_loss", out var vlEl))
        virtualLossValue = vlEl.GetDouble();
    if (mcts.TryGetProperty("lottery_eval_limit", out var lelEl))
        lotteryEvalLimit = lelEl.GetInt32();
    var sp = doc.RootElement.GetProperty("selfplay");
    parallelGames = sp.GetProperty("parallel_games").GetInt32();
    mctsThreads = sp.GetProperty("mcts_threads").GetInt32();
    dirichletAlpha = sp.GetProperty("dirichlet_alpha").GetDouble();
    dirichletEpsilon = sp.GetProperty("dirichlet_epsilon").GetDouble();
    maxMoves = sp.GetProperty("max_moves").GetInt32();
    if (sp.TryGetProperty("neural_batch_size", out var bsEl))
        neuralBatchSize = bsEl.GetInt32();
    if (sp.TryGetProperty("neural_batch_timeout_ms", out var btEl))
        neuralBatchTimeoutMs = btEl.GetInt32();
}
catch
{
    // config 读取失败就用默认值
}

Console.OutputEncoding = System.Text.Encoding.UTF8;

// ── 模式 2：对战评测（泛化：网络1/网络2/MCTS 组合）──
if (args.Length >= 1 && args[0] == "match")
{
    if (args.Length < 4)
    {
        Console.WriteLine("用法: SelfPlayCollector match <numGames> <net1|none> <outputDir> <progress> <pause> <prepare> [net2|none] [mcts2spec]");
        return;
    }
    int matchGames = int.Parse(args[1]);
    string net1 = args[2] == "none" ? null : args[2];
    string matchOutDir = Path.GetFullPath(args[3]);
    string matchProgress = args.Length > 4 ? args[4] : null;
    string matchPauseFlag = args.Length > 5 && args[5] != "-" && args[5] != "" ? args[5] : null;
    bool matchPrepare = args.Length > 6 && args[6] == "1";
    string net2 = args.Length > 7 && args[7] != "-" && args[7] != "" && File.Exists(args[7]) ? args[7] : null;
    if (net1 == "none" || (net1 != null && !File.Exists(net1))) net1 = null;
    string mcts2spec = args.Length > 8 ? args[8] : "";

    int matchParallel = parallelGames;
    try
    {
        using var doc2 = JsonDocument.Parse(File.ReadAllText(configPath));
        var sp2 = doc2.RootElement.GetProperty("selfplay");
        if (sp2.TryGetProperty("match_parallel_games", out var mpEl))
            matchParallel = mpEl.GetInt32();
    }
    catch { }
    // 对战模拟数固定等于全局 mcts.num_mcts_sims（公平性要求）
    MatchProgram.Run(matchGames, net1, net2, mcts2spec, matchOutDir, numSims, mctsThreads,
        maxMoves, cpuct, matchParallel, matchProgress, matchPrepare,
        evalMaterialWeight, virtualLossValue, lotteryEvalLimit, matchPauseFlag);
    return;
}

// ── 模式 1：自对弈数据收集 ──
if (args.Length < 2)
{
    Console.WriteLine("用法: SelfPlayCollector <numGames> <dataDir> [progressFile] [pauseFlag] [onnxPath]");
    return;
}

int numGames = int.Parse(args[0]);
string dataDir = Path.GetFullPath(args[1]);
string progressFile = args.Length > 2 ? args[2] : null;
string pauseFlag = args.Length > 3 ? args[3] : null;
string onnxPath = args.Length > 4 ? args[4] : null;

Console.WriteLine($"数据输出目录: {dataDir}");
Console.WriteLine($"每步模拟次数: {numSims}");
Console.WriteLine($"并行: {parallelGames} 局 × {mctsThreads} MCTS 线程" + (onnxPath != null ? $" | 批量推理(batch={neuralBatchSize})" : ""));
Console.WriteLine($"温度: {temperature}（前 {tempThreshold} 步）| Dirichlet: α={dirichletAlpha} ε={dirichletEpsilon} | cpuct={cpuct} | 最大步数: {maxMoves}");

SelfPlayTrainer.Run(numGames, numSims, mctsThreads, parallelGames, dataDir,
    progressFile, pauseFlag, onnxPath,
    dirichletAlpha, dirichletEpsilon, temperature, tempThreshold, cpuct, maxMoves,
    neuralBatchSize, neuralBatchTimeoutMs, evalMaterialWeight,
    virtualLossValue);
