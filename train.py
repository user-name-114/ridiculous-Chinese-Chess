import json
import os
import sys
import time
import struct
import glob
import torch
import torch.nn.functional as F
import numpy as np

from model import ChessNet, MOVE_SIZE, SNIPER_SIZE, LOTTERY_SIZE, TOTAL_ACTION_SIZE, INPUT_CH, BOARD_H, BOARD_W

# 强制 stdout 用 UTF-8 且绕过缓冲（write_through），
# 让 step 日志实时写入文件，面板才能实时读到进度和日志
try:
    sys.stdout.reconfigure(encoding="utf-8", write_through=True)
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 覆盖 print，默认 flush=True，确保每条日志立即落盘（不再积压到结束）
import builtins as _builtins
_orig_print = _builtins.print


def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)


_builtins.print = _flush_print

# ====================================================================
# 训练脚本：加载自对弈数据 → GPU 训练双头网络 → 定期保存 checkpoint。
#
# 支持：
#   1. 从 config.json 读取超参数
#   2. checkpoint 保存 / 续训（模型 + 优化器 + 步数 + 超参数）
#   3. 暂停标志（pause.flag 存在则保存后空转等待，删除后继续）
#
# 数据格式（.bin，由 C# SelfPlayTrainer 生成）：
#   [int32] num_samples
#   [int32] board_feature_size (3388)
#   [int32] graveyard_size (18)
#   每样本: board(3388f) + graveyard(18f) + num_actions(i)
#           + indices(n*i) + probs(n*f) + value(f)
#
# 策略标签索引范围 0~24332（移动 23716 + 狙击 616 + 抽奖 1 个标量）。
# 网络输出 24333 维，与训练标签直接对应。
# ====================================================================

PAUSE_FLAG = "pause.flag"
ROOT_ACTION_SIZE = MOVE_SIZE + SNIPER_SIZE + 1  # 24333（含抽奖标量）


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(data_dir):
    """读取 .bin 数据，并按文件（=按局）划分训练集与验证集（每20局划1局）。"""
    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not bin_files:
        print(f"[错误] {data_dir} 下没有 .bin 数据文件")
        return None

    tr_b, tr_g, tr_v = [], [], []
    tr_pol = []
    va_b, va_g, va_v = [], [], []
    va_pol = []
    val_every = 20

    for file_idx, fp in enumerate(bin_files):
        with open(fp, "rb") as f:
            raw = f.read()
        num_samples, board_size, grave_size = struct.unpack_from("iii", raw, 0)
        assert (board_size, grave_size) == (3388, 18), (fp, board_size, grave_size)

        boards = np.empty((num_samples, board_size), dtype=np.float32)
        graves = np.empty((num_samples, grave_size), dtype=np.float32)
        values = np.empty(num_samples, dtype=np.float32)
        pol_pairs = [None] * num_samples

        off = 12
        b_bytes = board_size * 4
        g_bytes = grave_size * 4
        f32 = np.float32
        i32 = np.int32
        for i in range(num_samples):
            boards[i] = np.frombuffer(raw, f32, board_size, off)
            off += b_bytes
            graves[i] = np.frombuffer(raw, f32, grave_size, off)
            off += g_bytes
            na = struct.unpack_from("i", raw, off)[0]
            off += 4
            idx = np.frombuffer(raw, i32, na, off)
            off += na * 4
            prob = np.frombuffer(raw, f32, na, off)
            off += na * 4
            values[i] = np.frombuffer(raw, f32, 1, off)[0]
            off += 4
            pol_pairs[i] = (idx.astype(np.int64, copy=False), prob.copy())

        if file_idx % val_every == 0:
            va_b.append(boards); va_g.append(graves); va_v.append(values)
            va_pol.extend(pol_pairs)
        else:
            tr_b.append(boards); tr_g.append(graves); tr_v.append(values)
            tr_pol.extend(pol_pairs)

    boards = np.concatenate(tr_b, axis=0).reshape(-1, INPUT_CH, BOARD_H, BOARD_W)
    graveyards = np.concatenate(tr_g, axis=0)
    values = np.concatenate(tr_v, axis=0)
    print(f"加载 {len(boards)} 个训练样本 + {sum(len(v) for v in va_v)} 个验证样本"
          f"（来自 {len(bin_files)} 个文件）")
    if va_b:
        v_boards = np.concatenate(va_b, axis=0).reshape(-1, INPUT_CH, BOARD_H, BOARD_W)
        v_graveyards = np.concatenate(va_g, axis=0)
        v_values = np.concatenate(va_v, axis=0)
    else:
        v_boards = np.empty((0, INPUT_CH, BOARD_H, BOARD_W), dtype=np.float32)
        v_graveyards = np.empty((0, 18), dtype=np.float32)
        v_values = np.empty((0,), dtype=np.float32)
    return boards, graveyards, tr_pol, values, v_boards, v_graveyards, va_pol, v_values



def sparse_cross_entropy(logits, indices_list, probs_list):
    """稀疏策略交叉熵（向量化版）。

    把整个 batch 的稀疏索引拼接后一次性从 log_softmax 里取值，
    不再逐样本 Python 循环（batch 大时循环是纯 GPU 空转）。
    """
    B = logits.size(0)
    log_probs = F.log_softmax(logits, dim=1)

    parts_idx = []
    parts_prob = []
    counts = []
    for idx, prob in zip(indices_list, probs_list):
        if idx.numel() == 0:
            continue
        parts_idx.append(idx)
        parts_prob.append(prob)
        counts.append(idx.numel())

    if not parts_idx:
        return logits.sum() * 0.0  # 空目标：保持图连通的零损失

    idx_flat = torch.cat(parts_idx, dim=0)
    prob_flat = torch.cat(parts_prob, dim=0)
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=logits.device)
    rows = torch.repeat_interleave(
        torch.arange(B, device=logits.device), counts_t)

    selected = log_probs[rows, idx_flat]
    return -(prob_flat * selected).sum() / B


def save_checkpoint(path, model, optimizer, step, config, elo=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "elo": elo,
    }, path)
    print(f"[checkpoint] step={step} → {path}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("[checkpoint] 该检查点无优化器状态（随机初始化网络），使用全新优化器")
    return ckpt.get("step", 0), ckpt.get("config", {})


def wait_while_paused(model, optimizer, step, config, checkpoint_dir, net_name="latest"):
    print("[暂停] 检测到 pause.flag，保存 checkpoint 并等待...")
    save_checkpoint(os.path.join(checkpoint_dir, f"{net_name}.pt"),
                    model, optimizer, step, config)
    while os.path.exists(PAUSE_FLAG):
        time.sleep(1.0)
    print("[继续] pause.flag 已删除，恢复训练")


def train(config, data, checkpoint_dir, resume_from=None, net_name="latest"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    net_cfg = config["network"]
    model = ChessNet(
        num_blocks=net_cfg["num_residual_blocks"],
        channels=net_cfg["channels"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if resume_from is not None and os.path.exists(resume_from):
        _, saved_cfg = load_checkpoint(resume_from, model, optimizer)
        if not saved_cfg:
            print("[checkpoint] 随机初始化网络：无保存配置，跳过结构比对（结构由当前 config 决定）")
        else:
            for key in ["num_residual_blocks", "channels"]:
                if saved_cfg["network"][key] != net_cfg[key]:
                    print(f"[错误] 结构超参数不一致: {key} "
                          f"(记录 {saved_cfg['network'][key]} vs 当前 {net_cfg[key]})")
                    sys.exit(1)
    boards, graveyards, policies, values, v_boards, v_graves, v_pols, v_vals = data
    boards_t = torch.from_numpy(boards).to(device)
    graves_t = torch.from_numpy(graveyards).to(device)
    values_t = torch.from_numpy(values).float().to(device)
    v_boards_t = torch.from_numpy(v_boards).to(device)
    v_graves_t = torch.from_numpy(v_graves).to(device)
    v_vals_t = torch.from_numpy(v_vals).float().to(device)
    # policies 是 list of (np indices, np probs)，转成 GPU tensor
    pol_idx = [torch.from_numpy(p[0]).long().to(device) for p in policies]
    pol_prob = [torch.from_numpy(p[1]).float().to(device) for p in policies]

    tr_cfg = config["training"]
    batch_size = tr_cfg["batch_size"]
    num_steps = tr_cfg["num_train_steps"]
    interval = tr_cfg["checkpoint_interval"]
    value_weight = tr_cfg["value_loss_weight"]
    use_amp = bool(tr_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB) | AMP={use_amp}")
    n = boards_t.size(0)

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    perm = np.random.permutation(n)  # 无放回抽样：打乱后的索引队列
    ptr = 0
    for step in range(num_steps):
        if os.path.exists(PAUSE_FLAG):
            wait_while_paused(model, optimizer, step, config, checkpoint_dir, net_name)

        # 无放回抽样：每个 epoch 内每个样本恰好用一次，用完重新打乱
        if ptr + batch_size > n:
            perm = np.random.permutation(n)
            ptr = 0
        indices = perm[ptr:ptr + batch_size]
        ptr += batch_size

        x = boards_t[indices]
        g = graves_t[indices]
        v_target = values_t[indices]
        b_idx = [pol_idx[i] for i in indices]
        b_prob = [pol_prob[i] for i in indices]

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                policy_logits, v_pred = model(x, g)
                v_pred = v_pred.squeeze(1)
                # BN/log_softmax 在 autocast 下自动跑 fp32，不担心精度问题
                policy_loss = sparse_cross_entropy(policy_logits, b_idx, b_prob)
                value_loss = F.mse_loss(v_pred.float(), v_target)
                loss = policy_loss + value_weight * value_loss
        else:
            policy_logits, v_pred = model(x, g)
            v_pred = v_pred.squeeze(1)
            policy_loss = sparse_cross_entropy(policy_logits, b_idx, b_prob)
            value_loss = F.mse_loss(v_pred, v_target)
            loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            with torch.no_grad():
                _es = sum(float(-(t * t.clamp_min(1e-12).log()).sum()) for t in b_prob)
                tgt_ent = _es / len(b_prob)
            print(f"step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"policy={policy_loss.item():.4f}（目标熵{tgt_ent:.4f}）  value={value_loss.item():.4f}")

        # 验证集评估（每 interval 步一次，纯前向，按局划分）
        if step > 0 and step % interval == 0 and v_boards.shape[0] > 0:
            model.eval()
            vps = vvs_ = 0.0
            vn = 0
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                for vstart in range(0, v_boards.shape[0], 256):
                    vslice = v_pols[vstart:vstart + 256]
                    vx = torch.from_numpy(v_boards[vstart:vstart + 256]).to(device)
                    vg = torch.from_numpy(v_graves[vstart:vstart + 256]).to(device)
                    vy = torch.from_numpy(v_vals[vstart:vstart + 256]).to(device)
                    vpol, vval = model(vx, vg)
                    vval = vval.squeeze(1)
                    vps += sparse_cross_entropy(vpol,
                                                [torch.from_numpy(t[0]).to(device) for t in vslice],
                                                [torch.from_numpy(t[1]).to(device) for t in vslice]).item()
                    vvs_ += F.mse_loss(vval.float(), vy).item()
                    vn += 1
            model.train()
            print(f"[验证] step {step + 1}: policy={vps / max(1, vn):.4f} value={vvs_ / max(1, vn):.4f}", flush=True)
        if step > 0 and step % interval == 0:
            save_checkpoint(os.path.join(checkpoint_dir, f"{net_name}.pt"),
                            model, optimizer, step, config)

    save_checkpoint(os.path.join(checkpoint_dir, f"{net_name}.pt"),
                    model, optimizer, num_steps, config)
    print("训练完成")


def main():
    config = load_config("config.json")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints"
    resume_from = sys.argv[3] if len(sys.argv) > 3 else None
    net_name = sys.argv[4] if len(sys.argv) > 4 else "latest"

    boards, graves, pols, vals, v_boards, v_graves, v_pols, v_vals = load_data(data_dir)
    if boards is None:
        return
    train(config, (boards, graves, pols, vals, v_boards, v_graves, v_pols, v_vals),
          checkpoint_dir, resume_from, net_name)


if __name__ == "__main__":
    main()
