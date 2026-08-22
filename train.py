import json
import os
import sys
import time
import struct
import glob
import torch
import torch.nn.functional as F
import numpy as np
import math

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
# 网络输出 27152 维，训练时前 24332 维直接对应标签，
# 后 2820 维（抽奖后续选择）log-sum-exp 聚合成 1 个"抽奖"标量。
# ====================================================================

PAUSE_FLAG = "pause.flag"
ROOT_ACTION_SIZE = MOVE_SIZE + SNIPER_SIZE + 1  # 24333（含抽奖标量）


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(data_dir):
    """加载 .bin 数据。返回 boards, graveyards, policies, values。"""
    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not bin_files:
        print(f"[错误] {data_dir} 下没有 .bin 数据文件")
        return None

    all_boards, all_graves, all_policies, all_values = [], [], [], []
    for fp in bin_files:
        with open(fp, "rb") as f:
            num_samples = struct.unpack("i", f.read(4))[0]
            board_size = struct.unpack("i", f.read(4))[0]
            grave_size = struct.unpack("i", f.read(4))[0]
            for _ in range(num_samples):
                board = struct.unpack(f"{board_size}f", f.read(board_size * 4))
                grave = struct.unpack(f"{grave_size}f", f.read(grave_size * 4))
                na = struct.unpack("i", f.read(4))[0]
                indices = struct.unpack(f"{na}i", f.read(na * 4))
                probs = struct.unpack(f"{na}f", f.read(na * 4))
                value = struct.unpack("f", f.read(4))[0]
                all_boards.append(board)
                all_graves.append(grave)
                all_policies.append((np.array(indices, dtype=np.int64),
                                     np.array(probs, dtype=np.float32)))
                all_values.append(value)

    boards = np.array(all_boards, dtype=np.float32)
    boards = boards.reshape(-1, INPUT_CH, BOARD_H, BOARD_W)  # (N, 22, 14, 11)
    graveyards = np.array(all_graves, dtype=np.float32)
    values = np.array(all_values, dtype=np.float32)
    print(f"加载 {len(boards)} 个样本（来自 {len(bin_files)} 个文件）")
    return boards, graveyards, all_policies, values


def aggregate_root_policy(policy_logits):
    """把网络 27152 维输出聚合成根节点 24333 维。

    抽奖段（2820 维）是「抽奖后选哪个格子/复活」的条件选择 logit。
    必须用 log-mean-exp（= log-sum-exp 减去 log(槽位数)）聚合成 1 个「抽奖」标量：
    - 不加修正的 log-sum-exp 会多出 log(2820)≈7.94 的固定偏差，让「抽奖」这个动作
      在 softmax 里被系统性高估（相当于抽奖凭空比每个移动动作高约 8 个 logit），
      训练几代后网络就会疯狂抽奖，形成正反馈。
    - 减去 log(2820) 后，「抽奖」的 logit 回到和「移动/狙击」动作同一尺度，
      反映的是「抽奖后平均能选到多好的目标」，这才是正确含义。
    """
    move = policy_logits[:, :MOVE_SIZE]
    sniper = policy_logits[:, MOVE_SIZE:MOVE_SIZE + SNIPER_SIZE]
    lottery_sub = policy_logits[:, MOVE_SIZE + SNIPER_SIZE:]  # 2820
    lottery_scalar = torch.logsumexp(lottery_sub, dim=1, keepdim=True) - math.log(LOTTERY_SIZE)
    return torch.cat([move, sniper, lottery_scalar], dim=1)  # (B, 24333)


def sparse_cross_entropy(logits, indices_list, probs_list):
    """稀疏策略交叉熵。logits (B, 24333)，indices/probs 是每样本的稀疏目标。"""
    log_probs = F.log_softmax(logits, dim=1)
    losses = []
    for i in range(logits.size(0)):
        idx = indices_list[i]
        prob = probs_list[i]
        losses.append(-torch.sum(prob * log_probs[i][idx]))
    return torch.stack(losses).mean()


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
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["config"]


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
        print("从 checkpoint 恢复（权重 + 优化器）")
        for key in ["num_residual_blocks", "channels"]:
            if saved_cfg["network"][key] != net_cfg[key]:
                print(f"[错误] 结构超参数不一致: {key} "
                      f"(记录 {saved_cfg['network'][key]} vs 当前 {net_cfg[key]})")
                sys.exit(1)

    boards, graveyards, policies, values = data
    boards_t = torch.from_numpy(boards).to(device)
    graves_t = torch.from_numpy(graveyards).to(device)
    values_t = torch.from_numpy(values).float().to(device)
    # policies 是 list of (np indices, np probs)，转成 GPU tensor
    pol_idx = [torch.from_numpy(p[0]).long().to(device) for p in policies]
    pol_prob = [torch.from_numpy(p[1]).float().to(device) for p in policies]

    tr_cfg = config["training"]
    batch_size = tr_cfg["batch_size"]
    num_steps = tr_cfg["num_train_steps"]
    interval = tr_cfg["checkpoint_interval"]
    value_weight = tr_cfg["value_loss_weight"]
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

        policy_logits, v_pred = model(x, g)
        v_pred = v_pred.squeeze(1)

        root_logits = aggregate_root_policy(policy_logits)  # (B, 24333)
        policy_loss = sparse_cross_entropy(root_logits, b_idx, b_prob)
        value_loss = F.mse_loss(v_pred, v_target)
        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"policy={policy_loss.item():.4f}  value={value_loss.item():.4f}")
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

    data = load_data(data_dir)
    if data is None:
        return
    train(config, data, checkpoint_dir, resume_from, net_name)


if __name__ == "__main__":
    main()
