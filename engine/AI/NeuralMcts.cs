using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// ====================================================================
// 神经网络增强 MCTS 的推理器。
//
// 加载训练好的 ONNX 模型（policy + value 双头），
// 把 Gamestate 编码成网络输入，前向得到：
//   - 根节点策略 logits（24333 维 = 移动 23716 + 狙击 616 + 抽奖 log-sum-exp 1）
//   - 局面价值 v（[-1, 1]，从当前玩家视角）
//
// 根节点策略的"抽奖"项由 2820 维抽奖后续选择的 log-sum-exp 聚合而来，
// 与 train.py 的 aggregate_root_policy 逻辑一致。
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
        catch (Exception ex)
        {
            var opts = new SessionOptions();
            opts.IntraOpNumThreads = 1; // batch=1 推理：单线程避免与大量 MCTS 线程竞争
            session = new InferenceSession(onnxPath, opts);
            Console.WriteLine("NeuralMcts: CUDA 不可用，回退 CPU 推理（单线程）");
        }
    }

    /// <summary>预测：输入局面，输出根节点策略 logits（24333 维）+ 价值</summary>
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

        // 聚合：27152 → 24333（抽奖后续选择 log-mean-exp 成 1 个标量）。
        // 关键：抽奖段是「抽奖后选目标」的条件 logit，必须减 log(2820)，
        // 否则 log-sum-exp 多出 log(2820) 偏差，让「抽奖」被系统性高估（疯狂抽奖）。
        float[] rootPolicy = new float[RootActionSize];
        int moveSniper = StateEncoder.MoveActionSize + StateEncoder.SniperActionSize;
        Array.Copy(policy, 0, rootPolicy, 0, moveSniper);

        float maxLottery = float.NegativeInfinity;
        for (int i = moveSniper; i < policy.Length; i++)
            if (policy[i] > maxLottery) maxLottery = policy[i];
        float sumExp = 0f;
        for (int i = moveSniper; i < policy.Length; i++)
            sumExp += (float)Math.Exp(policy[i] - maxLottery);
        int lotterySize = policy.Length - moveSniper; // 2820
        rootPolicy[RootActionSize - 1] = maxLottery + (float)Math.Log(sumExp) - (float)Math.Log(lotterySize);

        return (rootPolicy, value);
    }

    public void Dispose()
    {
        session?.Dispose();
    }
}
