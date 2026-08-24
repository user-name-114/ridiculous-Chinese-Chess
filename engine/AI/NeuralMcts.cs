using System;
using System.Collections.Generic;
using System.Linq;
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
// ====================================================================
public class NeuralMcts
{
    private InferenceSession session;
    private const int RootActionSize = 24333;  // 23716 + 616 + 1

    public NeuralMcts(string onnxPath)
    {
        AddCudaRuntimePaths();
        try
        {
            var opts = new SessionOptions();
            opts.AppendExecutionProvider_CUDA(0); // 优先用 GPU 0（RTX 4060）
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: 使用 CUDA (GPU) 推理");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"NeuralMcts: CUDA 初始化失败: {ex.Message}");
            var opts = new SessionOptions();
            opts.IntraOpNumThreads = 1; // batch=1 推理：单线程避免与大量 MCTS 线程竞争
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
            if (OperatingSystem.IsWindows())
                SetDllDirectory(candidate);
        }
        Environment.SetEnvironmentVariable("PATH", path);
    }

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    /// <summary>预测：输入局面，输出策略 logits（24333 维）和价值</summary>
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
                new DenseTensor<float>(graveyards,
                    new[] { batchSize, StateEncoder.GraveyardSize })),
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
            // 兼容动态 batch 修改前导出的旧模型
            float[] values = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
                values[i] = Predict(states[i]).value;
            return values;
        }
    }

    public void Dispose()
    {
        session?.Dispose();
    }
}
