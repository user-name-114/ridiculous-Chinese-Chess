using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

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
        try
        {
            var opts = new SessionOptions();
            opts.AppendExecutionProvider_CUDA(0); // 优先用 GPU 0（RTX 4060）
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: 使用 CUDA (GPU) 推理");
        }
        catch
        {
            var opts = new SessionOptions();
            opts.IntraOpNumThreads = 1; // batch=1 推理：单线程避免与大量 MCTS 线程竞争
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: CUDA 不可用，回退 CPU 推理（单线程）");
        }
    }

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

    public void Dispose()
    {
        session?.Dispose();
    }
}
