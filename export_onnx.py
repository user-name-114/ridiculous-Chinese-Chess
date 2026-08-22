import json
import os
import sys
import torch

from model import ChessNet, INPUT_CH, BOARD_H, BOARD_W, GRAVEYARD

# ====================================================================
# 把训练好的 PyTorch checkpoint (.pt) 导出成 ONNX 模型 (.onnx)。
#
# 用法: python export_onnx.py <checkpoint.pt> [output.onnx]
#   checkpoint.pt: train.py 训练出的 latest.pt
#   output.onnx:   导出路径（默认同目录 model.onnx）
#
# ONNX 输入:  board (1,22,14,11) + graveyard (1,18)
# ONNX 输出:  policy (1,27152) + value (1,1)
# ====================================================================

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export(checkpoint_path, onnx_path, config):
    net_cfg = config["network"]
    model = ChessNet(
        num_blocks=net_cfg["num_residual_blocks"],
        channels=net_cfg["channels"],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"已加载 checkpoint（step={ckpt.get('step', '?')}）")

    # 固定 batch=1（推理时单局面对弈）
    dummy_board = torch.randn(1, INPUT_CH, BOARD_H, BOARD_W)
    dummy_graveyard = torch.randn(1, GRAVEYARD)

    torch.onnx.export(
        model,
        (dummy_board, dummy_graveyard),
        onnx_path,
        input_names=["board", "graveyard"],
        output_names=["policy", "value"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,  # 用旧版导出器，避免依赖 onnxscript
    )
    print(f"ONNX 导出成功: {onnx_path}")
    print(f"  输入: board (1,{INPUT_CH},{BOARD_H},{BOARD_W}) + graveyard (1,{GRAVEYARD})")
    print(f"  输出: policy (1,27152) + value (1,1)")


def main():
    if len(sys.argv) < 2:
        print("用法: python export_onnx.py <checkpoint.pt> [output.onnx]")
        return
    checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        onnx_path = sys.argv[2]
    else:
        onnx_path = os.path.join(os.path.dirname(checkpoint_path), "model.onnx")
    config = load_config("config.json")
    export(checkpoint_path, onnx_path, config)


if __name__ == "__main__":
    main()
