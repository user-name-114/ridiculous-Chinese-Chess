import os
import shutil

# 把 Unity 项目里的纯 C# 引擎代码（MCTS + 规则 + 编码器）同步到训练文件夹
SRC = r"E:\chess\chess\Assets\Scripts\core"
DST = r"E:\chess\training\engine"

# 需要复制的文件（纯 C#，不含 Unity MonoBehaviour 的）
files = {
    # AI 文件夹
    os.path.join("AI", "AIPlayer.cs"): "AI",
    os.path.join("AI", "GameAction.cs"): "AI",
    os.path.join("AI", "LotteryResolver.cs"): "AI",
    os.path.join("AI", "MatchRunner.cs"): "AI",
    os.path.join("AI", "MctsEngine.cs"): "AI",
    os.path.join("AI", "MctsNode.cs"): "AI",
    os.path.join("AI", "StateEncoder.cs"): "AI",
    os.path.join("AI", "ActionEncoder.cs"): "AI",
    os.path.join("AI", "Benchmark.cs"): "AI",
    # GameLogic 文件夹（纯数据文件，排除 GameManager/Cell/SniperArrow 等 Unity 文件）
    os.path.join("GameLogic", "GameState.cs"): "GameLogic",
    os.path.join("GameLogic", "RuleEngine.cs"): "GameLogic",
    os.path.join("GameLogic", "PieceGenerator.cs"): "GameLogic",
    os.path.join("GameLogic", "LotteryManager.cs"): "GameLogic",
    os.path.join("GameLogic", "LotteryEffects.cs"): "GameLogic",
}

copied = []
for rel, sub in files.items():
    src = os.path.join(SRC, rel)
    dst_dir = os.path.join(DST, sub)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(rel))
    shutil.copy2(src, dst)
    copied.append(dst)

print(f"已复制 {len(copied)} 个文件到 {DST}:")
for c in copied:
    print("  " + c)
