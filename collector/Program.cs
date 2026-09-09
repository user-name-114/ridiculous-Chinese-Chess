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

// 读 config.json（两种模式共用）。
// 2026-09-08：加载失败/必需键缺失改为报错退出（旧版静默回退默认值
// sims=200/pg=8/tc=4/maxMoves=400/timeout=2，会让面板调参但 collector
// 实际没生效完全不可见）；启动时逐键回显完整路径+值。
// 2026-09-09：路径解析改为「当前工作目录优先，DLL 相对上溯 4 级回退」。
// 旧策略写死 DLL 相对路径，导致多 config 对照实验（cwd 放不同 config）
// 全部静默读主 config——A/B 实验结果无效（本轮抽奖 NN 评估实验首次踩中）。
// training 目录内正常运行时两种解析结果相同，无回归风险。
var configPath = File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "config.json"))
    ? Path.Combine(Directory.GetCurrentDirectory(), "config.json")
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "config.json"));
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
bool lotteryNnEval = false;
int neuralBatchSize = 32;
int neuralBatchTimeoutMs = 2; // ms
int minGameSamples = 10;

void LoadConfigStrict()
{
    void Fail(string msg)
    {
        Console.Error.WriteLine($"[config] 致命错误：{msg}");
        Console.Error.WriteLine($"[config] 解析路径 = {configPath}");
        Environment.Exit(1);
    }
    if (!File.Exists(configPath))
        Fail("找不到 config.json（优先按当前工作目录解析，回退 DLL 相对路径上溯 4 级；请从 training 目录结构内运行或在 cwd 放置 config.json）");
    JsonElement root;
    try
    {
        root = JsonDocument.Parse(File.ReadAllText(configPath)).RootElement;
    }
    catch (Exception ex)
    {
        Fail($"config.json 解析失败: {ex.Message}");
        return;
    }

    // 遗留同名键告警：防止新旧段并存导致读错段
    if (root.TryGetProperty("selfplay", out var spLegacy) &&
        spLegacy.TryGetProperty("num_mcts_sims", out _))
        Console.Error.WriteLine("[config] 警告：检测到遗留键 selfplay.num_mcts_sims（实际读取 mcts.num_mcts_sims），请删除该键");

    JsonElement GetSection(string name)
    {
        if (!root.TryGetProperty(name, out var s))
            Fail($"缺少必需段 [{name}]");
        return s;
    }
    JsonElement GetKey(JsonElement section, string key, string sectionName)
    {
        if (!section.TryGetProperty(key, out var v))
            Fail($"缺少必需键 [{sectionName}.{key}]");
        return v;
    }

    var mcts = GetSection("mcts");
    numSims       = GetKey(mcts, "num_mcts_sims", "mcts").GetInt32();
    temperature   = GetKey(mcts, "temperature", "mcts").GetDouble();
    tempThreshold = GetKey(mcts, "temp_threshold", "mcts").GetInt32();
    cpuct         = GetKey(mcts, "cpuct", "mcts").GetDouble();
    if (mcts.TryGetProperty("eval_material_weight", out var emwEl))
        evalMaterialWeight = emwEl.GetDouble();
    if (mcts.TryGetProperty("virtual_loss", out var vlEl))
        virtualLossValue = vlEl.GetDouble();
    if (mcts.TryGetProperty("lottery_eval_limit", out var lelEl))
        lotteryEvalLimit = lelEl.GetInt32();
    if (mcts.TryGetProperty("lottery_nn_eval", out var lneEl))
        lotteryNnEval = lneEl.GetBoolean();

    var sp = GetSection("selfplay");
    parallelGames    = GetKey(sp, "parallel_games", "selfplay").GetInt32();
    mctsThreads      = GetKey(sp, "mcts_threads", "selfplay").GetInt32();
    dirichletAlpha   = GetKey(sp, "dirichlet_alpha", "selfplay").GetDouble();
    dirichletEpsilon = GetKey(sp, "dirichlet_epsilon", "selfplay").GetDouble();
    maxMoves         = GetKey(sp, "max_moves", "selfplay").GetInt32();
    if (sp.TryGetProperty("neural_batch_size", out var bsEl))
        neuralBatchSize = bsEl.GetInt32();
    if (sp.TryGetProperty("neural_batch_timeout_ms", out var btEl))
        neuralBatchTimeoutMs = btEl.GetInt32();
    if (sp.TryGetProperty("min_game_samples", out var mgsEl))
        minGameSamples = mgsEl.GetInt32();

    // 启动回显：逐键打印完整路径 + 值
    Console.WriteLine($"[config] {configPath}");
    Console.WriteLine($"[config] mcts.num_mcts_sims = {numSims}");
    Console.WriteLine($"[config] mcts.temperature = {temperature}");
    Console.WriteLine($"[config] mcts.temp_threshold = {tempThreshold}");
    Console.WriteLine($"[config] mcts.cpuct = {cpuct}");
    Console.WriteLine($"[config] mcts.eval_material_weight = {evalMaterialWeight}");
    Console.WriteLine($"[config] mcts.virtual_loss = {virtualLossValue}");
    Console.WriteLine($"[config] mcts.lottery_eval_limit = {lotteryEvalLimit}");
    Console.WriteLine($"[config] mcts.lottery_nn_eval = {lotteryNnEval}");
    Console.WriteLine($"[config] selfplay.parallel_games = {parallelGames}");
    Console.WriteLine($"[config] selfplay.mcts_threads = {mctsThreads}");
    Console.WriteLine($"[config] selfplay.dirichlet_alpha = {dirichletAlpha}");
    Console.WriteLine($"[config] selfplay.dirichlet_epsilon = {dirichletEpsilon}");
    Console.WriteLine($"[config] selfplay.max_moves = {maxMoves}");
    Console.WriteLine($"[config] selfplay.neural_batch_size = {neuralBatchSize}");
    Console.WriteLine($"[config] selfplay.neural_batch_timeout_ms = {neuralBatchTimeoutMs}");
    Console.WriteLine($"[config] selfplay.min_game_samples = {minGameSamples}");
}
LoadConfigStrict();

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
        evalMaterialWeight, virtualLossValue, lotteryEvalLimit, matchPauseFlag,
        neuralBatchSize, neuralBatchTimeoutMs);
    return;
}

// ── 模式 1：自对弈数据收集 ──
if (args.Length < 2)
{
    Console.WriteLine("用法: SelfPlayCollector <numGames> <dataDir> [progressFile] [pauseFlag] [onnxPath|-] [prepareLottery:0|1]");
    return;
}

int numGames = int.Parse(args[0]);
string dataDir = Path.GetFullPath(args[1]);
string progressFile = args.Length > 2 ? args[2] : null;
string pauseFlag = args.Length > 3 ? args[3] : null;
// 2026-09-09：args[4] 支持 "-" 占位（无网络指导纯 MCTS），args[5] = 强制开局抽奖开关
string onnxPath = args.Length > 4 && args[4] != "-" ? args[4] : null;
bool prepareLottery = args.Length > 5 && args[5] == "1";

Console.WriteLine($"数据输出目录: {dataDir}");
Console.WriteLine($"每步模拟次数: {numSims}");
Console.WriteLine($"并行: {parallelGames} 局 × {mctsThreads} MCTS 线程" + (onnxPath != null ? $" | 批量推理(batch={neuralBatchSize})" : ""));
Console.WriteLine($"温度: {temperature}（前 {tempThreshold} 步）| Dirichlet: α={dirichletAlpha} ε={dirichletEpsilon} | cpuct={cpuct} | 最大步数: {maxMoves}");
Console.WriteLine($"开局准备(强制抽奖，不计入步数/样本): {prepareLottery}");

SelfPlayTrainer.Run(numGames, numSims, mctsThreads, parallelGames, dataDir,
    progressFile, pauseFlag, onnxPath,
    dirichletAlpha, dirichletEpsilon, temperature, tempThreshold, cpuct, maxMoves,
    neuralBatchSize, neuralBatchTimeoutMs, evalMaterialWeight,
    virtualLossValue, minGameSamples, lotteryNnEval, prepareLottery);
