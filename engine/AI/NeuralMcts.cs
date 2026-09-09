using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;
using System.Runtime.InteropServices;

// ====================================================================
// 神经网络增强 MCTS 的推理器。
//
// 加载训练好的 ONNX 模型（policy + value 双头），
// 把 Gamestate 编码成网络输入，前向得到策略 logits 和局面价值。
//
// 流水线批量推理服务（StartBatchService 后启用）：
//   收集线程：不停从请求队列收集，攒满或超时后放入 GPU 输入队列
//   GPU 线程：不停从 GPU 输入队列取批次，session.Run，分发结果
//   两者用 BlockingCollection 解耦——GPU 处理 batch N 时收集线程已在攒 batch N+1。
//
//   调用方线程在提交前自行完成 StateEncoder.Encode（分布在所有 MCTS 线程上），
//   收集线程只做 float[] 拼接，GPU 线程只做 session.Run。
// ====================================================================
public class NeuralMcts
{
    private InferenceSession session;
    private const int RootActionSize = 24333;  // 23716 + 616 + 1

    // ── 流水线批量推理 ──
    private BlockingCollection<PredictRequest> _requestQueue;  // MCTS线程 → 收集线程
    private BlockingCollection<BatchData> _gpuInputQueue;       // 收集线程 → GPU线程
    private Thread _collectorThread;
    private Thread _gpuThread;
    private volatile bool _batchRunning;
    private int _batchSize;
    private int _batchTimeoutMs;

    // 2026-09-04 GPU 线程统计（诊断用）：批数 / 样本总数 / GPU 计算累计 ticks
    internal static long StatGpuBatches, StatGpuSamples, StatGpuTicks;
    // 2026-09-05 T1~T4 子计时器：输入构造 / session.Run / D2H(ToArray) / 拆分循环
    internal static long StatT1, StatT2, StatT3, StatT4;

    // 调用方预编码后提交的请求
    private struct PredictRequest
    {
        public float[] Board;      // 预编码的棋盘特征 (3388)
        public float[] Graveyard;  // 预编码的墓地向量 (18)
        public TaskCompletionSource<(float[] policy, int policyOffset, float value)> Tcs;
    }

    // 收集线程攒好的批次，交给 GPU 线程
    private struct BatchData
    {
        public float[][] Boards;
        public float[][] Graveyards;
        public TaskCompletionSource<(float[] policy, int policyOffset, float value)>[] TcsList;
        public int Count;
    }


    public NeuralMcts(string onnxPath)
    {
        AddCudaRuntimePaths();
        try
        {
            var opts = new SessionOptions();
            opts.AppendExecutionProvider_CUDA(0);
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: 使用 CUDA (GPU) 推理");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"NeuralMcts: CUDA 初始化失败 {ex.Message}");
            var opts = new SessionOptions();
            opts.IntraOpNumThreads = 1;
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: CUDA 不可用，回退 CPU 推理（单线程）");
            // 2026-09-05 修复（外部审查指出）：CPU 静默回退会慢 ~30 倍且不易察觉——
            // 保留回退（不至于完全不能跑），但给出强信号：stderr 警告 + 进程退出码置 1，
            // 面板/脚本可通过退出码发现降级。
            Console.Error.WriteLine("[严重] NeuralMcts: 已回退 CPU 推理，速度约慢 30 倍！进程退出码已置 1。");
            Environment.ExitCode = 1;
        }
    }

    private static void AddCudaRuntimePaths()
    {
        var paths = new List<string>();
        string condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
        if (!string.IsNullOrEmpty(condaPrefix))
            paths.Add(Path.Combine(condaPrefix, "Lib", "site-packages", "torch", "lib"));
        paths.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "anaconda3", "Lib", "site-packages", "torch", "lib"));

        string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string candidate in paths)
        {
            if (!Directory.Exists(candidate)) continue;
            if (!path.Split(';').Contains(candidate, StringComparer.OrdinalIgnoreCase))
                path = candidate + ";" + path;
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN || PLATFORM_STANDALONE_WIN
            SetDllDirectory(candidate);
#endif
        }
        Environment.SetEnvironmentVariable("PATH", path);
    }

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    // ================================================================
    //  同步推理（游戏内 AI 使用，或批量服务未启动时回退）
    // ================================================================

    public (float[] rootPolicy, float value) Predict(Gamestate state)
    {
        float[] board = StateEncoder.Encode(state);
        float[] graveyard = StateEncoder.EncodeGraveyard(state);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("board",
                new DenseTensor<float>(board,
                    new[] { 1, StateEncoder.Channels, StateEncoder.Height, StateEncoder.Width })),
            NamedOnnxValue.CreateFromTensor("graveyard",
                new DenseTensor<float>(graveyard, new[] { 1, StateEncoder.GraveyardSize })),
        };

        using var results = session.Run(inputs);
        float[] policy = results.First(r => r.Name == "policy").AsTensor<float>().ToArray();
        float value = results.First(r => r.Name == "value").AsTensor<float>().ToArray()[0];

        if (policy.Length != RootActionSize)
            throw new InvalidOperationException($"策略输出维度错误: {policy.Length}，期望 {RootActionSize}");

        return (policy, value);
    }

    // ================================================================
    //  流水线批量推理服务
    // ================================================================

    public void StartBatchService(int batchSize = 32, int batchTimeoutMs = 2)
    {
        _batchSize = batchSize;
        _batchTimeoutMs = batchTimeoutMs;
        _requestQueue = new BlockingCollection<PredictRequest>(8192);
        _gpuInputQueue = new BlockingCollection<BatchData>(64);
        _batchRunning = true;

        _collectorThread = new Thread(CollectorLoop)
        {
            IsBackground = true,
            Name = "NeuralCollector",
            Priority = ThreadPriority.AboveNormal
        };
        _gpuThread = new Thread(GpuLoop)
        {
            IsBackground = true,
            Name = "NeuralGpu",
            Priority = ThreadPriority.AboveNormal
        };
        _collectorThread.Start();
        _gpuThread.Start();
        Console.WriteLine($"NeuralMcts: 流水线批量推理已启动 (batch={batchSize}, timeout={batchTimeoutMs}ms)");
    }

    public void StopBatchService()
    {
        if (!_batchRunning) return;
        _batchRunning = false;
        _requestQueue?.CompleteAdding();
        _gpuInputQueue?.CompleteAdding();
        _collectorThread?.Join(10000);
        _gpuThread?.Join(10000);
        _requestQueue?.Dispose();
        _gpuInputQueue?.Dispose();
        _requestQueue = null;
        _gpuInputQueue = null;
        _collectorThread = null;
        _gpuThread = null;
        Console.WriteLine("NeuralMcts: 流水线批量推理已停止");
    }

    /// <summary>提交推理请求，阻塞直到结果返回。
    /// 调用方线程在提交前完成 StateEncoder.Encode。
    /// 2026-09-07 零拷贝：policy 为整批共享的底层缓冲，policyOffset 是本样本起始下标。</summary>
    public (float[] policy, int policyOffset, float value) PredictBlocking(Gamestate state)
    {
        if (_requestQueue == null || !_batchRunning)
        {
            var (p, v) = Predict(state);
            return (p, 0, v);
        }

        float[] board = StateEncoder.Encode(state);
        float[] graveyard = StateEncoder.EncodeGraveyard(state);

        var tcs = new TaskCompletionSource<(float[] policy, int policyOffset, float value)>(TaskCreationOptions.RunContinuationsAsynchronously);   // 2026-09-04 修复：避免 GPU 线程在 SetResult 时同步跑 worker continuation
        _requestQueue.Add(new PredictRequest { Board = board, Graveyard = graveyard, Tcs = tcs });
        return tcs.Task.Result;
    }

    // ── 收集线程：从请求队列攒批，放入 GPU 输入队列 ──

    private void CollectorLoop()
    {
        while (_batchRunning)
        {
            var boards = new List<float[]>(_batchSize);
            var graveyards = new List<float[]>(_batchSize);
            var tcsList = new List<TaskCompletionSource<(float[] policy, int policyOffset, float value)>>(_batchSize);

            // 阻塞等第一个请求
            PredictRequest first;
            try { first = _requestQueue.Take(); }
            catch (InvalidOperationException) { break; }

            boards.Add(first.Board);
            graveyards.Add(first.Graveyard);
            tcsList.Add(first.Tcs);

            // 尽量收集更多填满批次
            while (boards.Count < _batchSize)
            {
                if (_requestQueue.TryTake(out var item, _batchTimeoutMs))
                {
                    boards.Add(item.Board);
                    graveyards.Add(item.Graveyard);
                    tcsList.Add(item.Tcs);
                }
                else
                    break;
            }

            // 放入 GPU 队列（GPU 线程可能在等这一批）
            try
            {
                _gpuInputQueue.Add(new BatchData
                {
                    Boards = boards.ToArray(),
                    Graveyards = graveyards.ToArray(),
                    TcsList = tcsList.ToArray(),
                    Count = boards.Count
                });
            }
            catch (InvalidOperationException) { break; }
        }
    }

    // ── GPU 线程：取批次 → session.Run → 分发结果 ──

    private void GpuLoop()
    {
        while (_batchRunning)
        {
            BatchData batch;
            try { batch = _gpuInputQueue.Take(); }
            catch (InvalidOperationException) { break; }

            System.Threading.Interlocked.Increment(ref StatGpuBatches);
            System.Threading.Interlocked.Add(ref StatGpuSamples, batch.Count);
            var __g = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var (flatPolicy, values) = PredictBatchFromEncoded(
                    batch.Boards, batch.Graveyards, batch.Count);
                for (int i = 0; i < batch.Count; i++)
                    batch.TcsList[i].SetResult((flatPolicy, i * RootActionSize, values[i]));
            }
            catch (Exception ex)
            {
                for (int i = 0; i < batch.Count; i++)
                    batch.TcsList[i].SetException(ex);
            }
            finally
            {
                System.Threading.Interlocked.Add(ref StatGpuTicks, __g.ElapsedTicks);
            }
        }
    }

    private (float[] flatPolicy, float[] values) PredictBatchFromEncoded(
        float[][] boards, float[][] graveyards, int batch)
    {
        float[] boardFlat = new float[batch * StateEncoder.FeatureSize];
        float[] graveFlat = new float[batch * StateEncoder.GraveyardSize];

        for (int i = 0; i < batch; i++)
        {
            Array.Copy(boards[i], 0, boardFlat, i * StateEncoder.FeatureSize, StateEncoder.FeatureSize);
            Array.Copy(graveyards[i], 0, graveFlat, i * StateEncoder.GraveyardSize, StateEncoder.GraveyardSize);
        }

        var __t = System.Diagnostics.Stopwatch.StartNew();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("board",
                new DenseTensor<float>(boardFlat,
                    new[] { batch, StateEncoder.Channels, StateEncoder.Height, StateEncoder.Width })),
            NamedOnnxValue.CreateFromTensor("graveyard",
                new DenseTensor<float>(graveFlat, new[] { batch, StateEncoder.GraveyardSize })),
        };
        System.Threading.Interlocked.Add(ref StatT1, __t.ElapsedTicks);
        __t.Restart();

        using var results = session.Run(inputs);
        System.Threading.Interlocked.Add(ref StatT2, __t.ElapsedTicks);
        __t.Restart();

        // 2026-09-07 零拷贝优化（T3+T4 从 ~4.9ms/批 → 近零）：
        // 旧实现 policyFlat = ...ToArray() 多复制一次 ~3MB，再逐样本 new float[24333]
        // 拆分（31×97KB 的 LOH 分配是 T4 异常 3.17ms 的主因）。
        // 现在直接把 ORT 输出的底层缓冲按偏移交给 worker（视图共享同一数组，
        // 数组随 worker 引用存活，无覆盖风险），数值与旧路径逐位一致。
        var policyTensor = results.First(r => r.Name == "policy").AsTensor<float>();
        var valueTensor = results.First(r => r.Name == "value").AsTensor<float>();
        if (policyTensor is DenseTensor<float> pdt
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(pdt.Buffer, out var pseg)
            && pseg.Offset == 0 && pseg.Count == batch * RootActionSize
            && valueTensor is DenseTensor<float> vdt
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(vdt.Buffer, out var vseg)
            && vseg.Count == batch)
        {
            System.Threading.Interlocked.Add(ref StatT3, __t.ElapsedTicks);
            __t.Restart();
            System.Threading.Interlocked.Add(ref StatT4, __t.ElapsedTicks);
            return (pseg.Array, vseg.Array);
        }

        // 兜底：底层缓冲不可直接提取时走旧拷贝路径
        float[] policyFlat = policyTensor.ToArray();
        float[] values = valueTensor.ToArray();
        System.Threading.Interlocked.Add(ref StatT3, __t.ElapsedTicks);
        __t.Restart();
        System.Threading.Interlocked.Add(ref StatT4, __t.ElapsedTicks);
        return (policyFlat, values);
    }

    // ================================================================
    //  批量价值评估（抽奖候选选择，直接调用，不走批量队列）
    // ================================================================

    public float[] PredictValues(IReadOnlyList<Gamestate> states)
    {
        if (states.Count == 0)
            return Array.Empty<float>();
        if (states.Count == 1)
            return new[] { Predict(states[0]).value };

        int batchSize = states.Count;
        float[] boards = new float[batchSize * StateEncoder.FeatureSize];
        float[] graveyards = new float[batchSize * StateEncoder.GraveyardSize];
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(StateEncoder.Encode(states[i]), 0, boards,
                i * StateEncoder.FeatureSize, StateEncoder.FeatureSize);
            Array.Copy(StateEncoder.EncodeGraveyard(states[i]), 0, graveyards,
                i * StateEncoder.GraveyardSize, StateEncoder.GraveyardSize);
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("board",
                new DenseTensor<float>(boards,
                    new[] { batchSize, StateEncoder.Channels, StateEncoder.Height, StateEncoder.Width })),
            NamedOnnxValue.CreateFromTensor("graveyard",
                new DenseTensor<float>(graveyards, new[] { batchSize, StateEncoder.GraveyardSize })),
        };

        try
        {
            using var results = session.Run(inputs);
            float[] values = results.First(r => r.Name == "value").AsTensor<float>().ToArray();
            if (values.Length != batchSize)
                throw new InvalidOperationException($"价值输出维度错误: {values.Length}，期望 {batchSize}");
            return values;
        }
        catch (OnnxRuntimeException)
        {
            float[] values = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
                values[i] = Predict(states[i]).value;
            return values;
        }
    }

    // ================================================================
    //  批量价值评估 v2（2026-09-09，抽奖候选选择：走共享请求队列与主搜索合批）
    // ================================================================

    /// <summary>预编码批量价值评估：N 个请求塞进现有请求队列（与主搜索共用批量流水线，
    /// 天然合批、不与 GPU 争抢），阻塞等待全部返回后取 value。
    /// 替代旧版独立同步 PredictValues（2026-09-08 实测 +301% 的根因是独立 session.Run
    /// 与主流水线争抢 GPU）。批量服务未启动时退化为单次同步批量前向。
    /// 调用方需预先完成 StateEncoder.Encode/EncodeGraveyard（形状与叶子节点完全一致）。</summary>
    public float[] PredictValuesViaQueue(float[][] boards, float[][] graveyards)
    {
        int n = boards.Length;
        if (n == 0) return Array.Empty<float>();

        if (_requestQueue == null || !_batchRunning)
        {
            var (_, syncValues) = PredictBatchFromEncoded(boards, graveyards, n);
            return syncValues;
        }

        var tcsList = new TaskCompletionSource<(float[] policy, int policyOffset, float value)>[n];
        var tasks = new Task[n];
        for (int i = 0; i < n; i++)
        {
            tcsList[i] = new TaskCompletionSource<(float[] policy, int policyOffset, float value)>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            tasks[i] = tcsList[i].Task;
            _requestQueue.Add(new PredictRequest { Board = boards[i], Graveyard = graveyards[i], Tcs = tcsList[i] });
        }
        Task.WaitAll(tasks);
        var values = new float[n];
        for (int i = 0; i < n; i++)
            values[i] = tcsList[i].Task.Result.value;
        return values;
    }

    public void Dispose()
    {
        StopBatchService();
        session?.Dispose();
    }
}
