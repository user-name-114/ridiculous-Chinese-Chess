# -*- coding: utf-8 -*-
"""生成随机初始化网络（AlphaZero 式冷启动基线）。

用法: python random_init_net.py <seed> <out_dir> <net_name>
输出: <out_dir>/<net_name>.pt + <net_name>.onnx + init_info.txt + init_run_log.txt
由训练界面"随机初始化网络"按钮调用；也可手动运行。
"""
import io, os, sys, json, random, datetime
import numpy as np

if sys.stdout is None:      # pythonw 无控制台时回退
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import ChessNet

FEATURE_NOTE = "board(1,22,14,11) + graveyard(1,18)"
_log_fh = None
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_run_log.txt')

def log(msg):
    print(msg)
    try:
        global _log_fh
        if _log_fh is None:
            _log_fh = open(_log_path, 'w', encoding='utf-8')
        _log_fh.write(msg + '\n'); _log_fh.flush()
    except Exception:
        pass

def main():
    global _log_path
    if len(sys.argv) < 4:
        print("用法: python random_init_net.py <seed> <out_dir> <net_name>")
        return 1
    seed = int(sys.argv[1]); out_dir = sys.argv[2]; name = sys.argv[3]
    _log_path = os.path.join(out_dir, 'init_run_log.txt')
    full = json.load(open("config.json", encoding="utf-8"))
    cfg = full["network"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    net = ChessNet(num_blocks=cfg["num_residual_blocks"], channels=cfg["channels"])

    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, name + ".pt")
    n_params = sum(p.numel() for p in net.parameters())
    torch.save({"model_state_dict": net.state_dict(), "step": 0, "seed": seed,
                "init": "random"}, pt_path)
    log(f"[随机初始化] 种子={seed}")
    log(f"[随机初始化] 结构: {cfg['num_residual_blocks']} 块 × {cfg['channels']} 通道 | 参数量={n_params:,}")
    log(f"[随机初始化] 已保存: {pt_path}")

    import export_onnx
    onnx_path = os.path.join(out_dir, name + ".onnx")
    export_onnx.export(pt_path, onnx_path, full)
    log(f"[随机初始化] ONNX 导出成功: {onnx_path}")

    with open(os.path.join(out_dir, "init_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"随机初始化网络: {name}\n")
        f.write(f"种子: {seed}\n")
        f.write(f"结构: {cfg['num_residual_blocks']} 块 × {cfg['channels']} 通道\n")
        f.write(f"参数量: {n_params:,}\n")
        f.write("初始化方式: torch 默认随机初始化（种子可控，可复现）\n")
        f.write(f"时间: {datetime.datetime.now()}\n")
        f.write("用途: 可直接用于\"网络指导自对弈\"或作为对照基线\n")
    log("[随机初始化] init_info.txt 已写入（种子与参数）")
    return 0

if __name__ == "__main__":
    sys.exit(main())
