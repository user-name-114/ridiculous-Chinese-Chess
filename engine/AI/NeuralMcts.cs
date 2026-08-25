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

    // 调用方预编码后提交的请求
    private struct PredictRequest
    {
        public float[] Board;      // 预编码的棋盘特征 (3388)
        public float[] Graveyard;  // 预编码的墓地向量 (18)
        public TaskCompletionSource<(float[] policy, float value)> Tcs;
    }

    // 收集线程攒好的批次，交给 GPU 线程
    private struct BatchData
    {
        public float[][] Boards;
        public float[][] Graveyards;
        public TaskCompletionSource<(float[] policy, float value)>[] TcsList;
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
            Console.WriteLine($"NeuralMcts: CUDA 初始化失败: {ex.Message}");
            var opts = new SessionOptions();
            opts.IntraOpNumThreads = 1;
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: CUDA 不可用，回退 CPU 推理（单线程）");
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
    /// 调用方线程在提交前完成 StateEncoder.Encode。</summary>
    public (float[] policy, float value) PredictBlocking(Gamestate state)
    {
        if (_requestQueue == null || !_batchRunning)
            return Predict(state);

        float[] board = StateEncoder.Encode(state);
        float[] graveyard = StateEncoder.EncodeGraveyard(state);

        var tcs = new TaskCompletionSource<(float[] policy, float value)>();
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
            var tcsList = new List<TaskCompletionSource<(float[] policy, float value)>>(_batchSize);

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

            try
            {
                var (policies, values) = PredictBatchFromEncoded(
                    batch.Boards, batch.Graveyards, batch.Count);
                for (int i = 0; i < batch.Count; i++)
                    batch.TcsList[i].SetResult((policies[i], values[i]));
            }
            catch (Exception ex)
            {
                for (int i = 0; i < batch.Count; i++)
                    batch.TcsList[i].SetException(ex);
            }
        }
    }

    private (float[][] policies, float[] values) PredictBatchFromEncoded(
        float[][] boards, float[][] graveyards, int batch)
    {
        float[] boardFlat = new float[batch * StateEncoder.FeatureSize];
        float[] graveFlat = new float[batch * StateEncoder.GraveyardSize];

        for (int i = 0; i < batch; i++)
        {
            Array.Copy(boards[i], 0, boardFlat, i * StateEncoder.FeatureSize, StateEncoder.FeatureSize);
            Array.Copy(graveyards[i], 0, graveFlat, i * StateEncoder.GraveyardSize, StateEncoder.GraveyardSize);
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("board",
                new DenseTensor<float>(boardFlat,
                    new[] { batch, StateEncoder.Channels, StateEncoder.Height, StateEncoder.Width })),
            NamedOnnxValue.CreateFromTensor("graveyard",
                new DenseTensor<float>(graveFlat, new[] { batch, StateEncoder.GraveyardSize })),
        };

        using var results = session.Run(inputs);
        float[] policyFlat = results.First(r => r.Name == "policy").AsTensor<float>().ToArray();
        float[] values = results.First(r => r.Name == "value").AsTensor<float>().ToArray();

        float[][] policies = new float[batch][];
        for (int i = 0; i < batch; i++)
        {
            policies[i] = new float[RootActionSize];
            Array.Copy(policyFlat, i * RootActionSize, policies[i], 0, RootActionSize);
        }

        return (policies, values);
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

    public void Dispose()
    {
        StopBatchService();
        session?.Dispose();
    }
}
