import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 OpenMP 运行时冲突在退出时中止进程
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


def pad_legal(leg_lists):
    """将每样本的合法动作索引列表填充为 (N, L_max) int64 矩阵 + 长度数组。"""
    n = len(leg_lists)
    if n == 0:
        return np.empty((0, 1), dtype=np.int64), np.empty(0, dtype=np.int64)
    lm = max(len(x) for x in leg_lists)
    m = np.zeros((n, lm), dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    for i, x in enumerate(leg_lists):
        if len(x):
            m[i, :len(x)] = x
            lens[i] = len(x)
    return m, lens


def load_data(data_dir):
    """读取 .bin 数据，并按文件（=按局）划分训练集与验证集（每20局划1局）。
    支持 v1（无掩码）与 v2（含合法动作掩码）两种格式，按文件自动探测；
    同一目录混用版本会报错（面板逐代采集天然同质）。"""
    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not bin_files:
        print(f"[错误] {data_dir} 下没有 .bin 数据文件")
        return None

    tr_b, tr_g, tr_v = [], [], []
    tr_pol, tr_legal = [], []
    va_b, va_g, va_v = [], [], []
    va_pol, va_legal = [], []
    val_every = 20
    version = 1

    for file_idx, fp in enumerate(bin_files):
        with open(fp, "rb") as f:
            raw = f.read()
        num_samples, board_size, grave_size = struct.unpack_from("iii", raw, 0)
        assert (board_size, grave_size) == (3388, 18), (fp, board_size, grave_size)
        # 版本探测：v2 在头后多一个 int32=2；v1 的该位置是首样本 board
        # 首浮点（恒 0.0，其 int 位模式为 0），探测无歧义
        ver = 1
        off = 12
        if len(raw) >= 16:
            v = struct.unpack_from("i", raw, off)[0]
            if v == 2:
                ver = 2
                off += 4
        if file_idx == 0:
            version = ver
        elif ver != version:
            raise RuntimeError(f"目录内 .bin 版本不一致: {fp} 是 v{ver}，"
                               f"其他是 v{version}（请分开存放或重新采集）")

        boards = np.empty((num_samples, board_size), dtype=np.float32)
        graves = np.empty((num_samples, grave_size), dtype=np.float32)
        values = np.empty(num_samples, dtype=np.float32)
        pol_pairs = [None] * num_samples
        leg_lists = [None] * num_samples

        b_bytes = board_size * 4
        g_bytes = grave_size * 4
        f32 = np.float32
        i32 = np.int32
        for i in range(num_samples):
            boards[i] = np.frombuffer(raw, f32, board_size, off)
            off += b_bytes
            graves[i] = np.frombuffer(raw, f32, grave_size, off)
            off += g_bytes
            if version == 2:
                nL = struct.unpack_from("i", raw, off)[0]
                off += 4
                legal = np.frombuffer(raw, i32, nL, off).astype(np.int64, copy=False)
                off += nL * 4
            else:
                legal = None
            na = struct.unpack_from("i", raw, off)[0]
            off += 4
            idx = np.frombuffer(raw, i32, na, off)
            off += na * 4
            prob = np.frombuffer(raw, f32, na, off)
            off += na * 4
            values[i] = np.frombuffer(raw, f32, 1, off)[0]
            off += 4
            pol_pairs[i] = (idx.astype(np.int64, copy=False), prob.copy())
            leg_lists[i] = legal

        if file_idx % val_every == 0:
            va_b.append(boards); va_g.append(graves); va_v.append(values)
            va_pol.extend(pol_pairs); va_legal.extend(leg_lists)
        else:
            tr_b.append(boards); tr_g.append(graves); tr_v.append(values)
            tr_pol.extend(pol_pairs); tr_legal.extend(leg_lists)

    boards = np.concatenate(tr_b, axis=0).reshape(-1, INPUT_CH, BOARD_H, BOARD_W)
    graveyards = np.concatenate(tr_g, axis=0)
    values = np.concatenate(tr_v, axis=0)
    print(f"加载 {len(boards)} 个训练样本 + {sum(len(v) for v in va_v)} 个验证样本"
          f"（来自 {len(bin_files)} 个文件，格式 v{version}）")
    if va_b:
        v_boards = np.concatenate(va_b, axis=0).reshape(-1, INPUT_CH, BOARD_H, BOARD_W)
        v_graveyards = np.concatenate(va_g, axis=0)
        v_values = np.concatenate(va_v, axis=0)
    else:
        v_boards = np.empty((0, INPUT_CH, BOARD_H, BOARD_W), dtype=np.float32)
        v_graveyards = np.empty((0, 18), dtype=np.float32)
        v_values = np.empty((0,), dtype=np.float32)
    if version == 2:
        tr_legal, tr_legal_len = pad_legal(tr_legal)
        va_legal, va_legal_len = pad_legal(va_legal)
    else:
        tr_legal = tr_legal_len = va_legal = va_legal_len = None
    return (boards, graveyards, tr_pol, values, v_boards, v_graveyards, va_pol, v_values,
            version, tr_legal, tr_legal_len, va_legal, va_legal_len)



def sparse_cross_entropy(logits, idx_mat, prob_mat):
    """稠密 padded 版稀疏策略交叉熵（2026-09-04，v1 全 softmax 路径）。

    idx_mat/prob_mat: (B, K_max)，无效位置 prob=0（idx 填 0，乘 0 后不产生贡献）。
    与旧稀疏拼接版数学等价：-(Σ prob·log_softmax(logits)[idx]).sum()/B。
    """
    B = logits.size(0)
    log_probs = F.log_softmax(logits, dim=1)
    selected = log_probs.gather(1, idx_mat)
    return -(selected * prob_mat).sum(dim=1).mean()


def policy_cross_entropy(policy_logits, pos_mat, prob_mat,
                         leg_idx=None, leg_len=None, version=1):
    """策略损失（2026-09-08 v2 掩码路径）。

    version>=2：先 gather 出合法动作 logits（v2 的 pos_mat 是目标在
    合法列表内的位置），在合法集合内 log_softmax（合法数不足 L_max
    的行用 -1e9 屏蔽 padding），再按 pos 取目标项——非法动作不再
    进入分母（回收 ~13% 的梯度浪费，推理/MCTS 行为不变）。
    version=1：回退全 softmax 稀疏 CE（兼容旧数据）。"""
    if version < 2 or leg_idx is None:
        return sparse_cross_entropy(policy_logits, pos_mat, prob_mat)
    L = leg_idx.size(1)
    # AMP 下 policy_logits 可能是 fp16：先升 fp32，避免 -1e9 掩码值溢出
    legal_logits = policy_logits.gather(1, leg_idx).float()
    len_mask = torch.arange(L, device=policy_logits.device).unsqueeze(0) < leg_len.unsqueeze(1)
    legal_logits = legal_logits.masked_fill(~len_mask, -1e9)
    logp = F.log_softmax(legal_logits, dim=1)
    selected = logp.gather(1, pos_mat)
    return -(prob_mat * selected).sum(dim=1).mean()

def save_checkpoint(path, model, optimizer, step, config, elo=None,
                    val_policy_loss=None, val_value_loss=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "elo": elo,
        # 2026-09-08：每个验证点存档自带当次验证损失，供训练后对比选优
        "val_policy_loss": val_policy_loss,
        "val_value_loss": val_value_loss,
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
            # 价值头接管位置：旧检查点无该字段 → 视为接在全部残差块后
            saved_tap = saved_cfg["network"].get(
                "value_tap_block", saved_cfg["network"]["num_residual_blocks"])
            cur_tap = net_cfg.get("value_tap_block", net_cfg["num_residual_blocks"])
            if saved_tap != cur_tap:
                print(f"[错误] 结构超参数不一致: value_tap_block "
                      f"(记录 {saved_tap} vs 当前 {cur_tap})")
                sys.exit(1)
    boards, graveyards, policies, values, v_boards, v_graves, v_pols, v_vals, version, leg_idx_np, leg_len_np, v_leg_idx_np, v_leg_len_np = data
    # 修复（2026-09-04）：数据集不再全量常驻 GPU——旧版占显存约62%且随num_games线性增长。
    # 改为 CPU numpy 存储 + 每 step 按 batch 索引后单独传 GPU；
    # policies 打包为 (N, K_max) 稠密矩阵（无效位 prob=0），与稀疏拼接版数学严格等价。
    # v2（掩码）：policies 的 idx 是目标在合法列表内的位置，
    # leg_idx/leg_len 提供每样本的合法动作集合。
    boards_np, graves_np, values_np = boards, graveyards, values
    N = len(policies)
    K = max((len(p[0]) for p in policies), default=1)
    pol_idx_np = np.zeros((N, K), dtype=np.int64)
    pol_prob_np = np.zeros((N, K), dtype=np.float32)
    for i, p in enumerate(policies):
        k = len(p[0])
        pol_idx_np[i, :k] = p[0]
        pol_prob_np[i, :k] = p[1]
    Nv = len(v_pols)
    Kv = max((len(t[0]) for t in v_pols), default=1)
    v_pol_idx_np = np.zeros((Nv, Kv), dtype=np.int64)
    v_pol_prob_np = np.zeros((Nv, Kv), dtype=np.float32)
    for i, t in enumerate(v_pols):
        k = len(t[0])
        v_pol_idx_np[i, :k] = t[0]
        v_pol_prob_np[i, :k] = t[1]
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
    n = boards_np.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(42); np.random.seed(42)
    model.train()
    # ── 需求 1/2/3/4/5/6 状态跟踪 ──
    val_policy_hist = []   # [(步数, 验证 policy loss)]
    val_value_hist = []    # [(步数, 验证 value loss)]
    ckpts = []             # [(步数, 路径, 验证 policy loss, 验证 value loss)]
    # 2026-09-09（用户要求）：value loss 早停机制已取消——训练始终跑满 num_train_steps，
    # 主/副版本由训练后处理按 value<1 筛选规则决定。
    last_val_step = None
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

        x = torch.from_numpy(boards_np[indices]).to(device, non_blocking=True)
        g = torch.from_numpy(graves_np[indices]).to(device, non_blocking=True)
        v_target = torch.from_numpy(values_np[indices]).float().to(device, non_blocking=True)
        b_idx = torch.from_numpy(pol_idx_np[indices]).to(device, non_blocking=True)
        b_prob = torch.from_numpy(pol_prob_np[indices]).to(device, non_blocking=True)
        b_leg = (torch.from_numpy(leg_idx_np[indices]).to(device, non_blocking=True)
                 if leg_idx_np is not None else None)
        b_leglen = (torch.from_numpy(leg_len_np[indices]).to(device, non_blocking=True)
                    if leg_len_np is not None else None)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                policy_logits, v_pred = model(x, g)
                v_pred = v_pred.squeeze(1)
                # BN/log_softmax 在 autocast 下自动跑 fp32，不担心精度问题
                policy_loss = policy_cross_entropy(policy_logits, b_idx, b_prob,
                                                   b_leg, b_leglen, version)
                value_loss = F.mse_loss(v_pred.float(), v_target)
                loss = policy_loss + value_weight * value_loss
        else:
            policy_logits, v_pred = model(x, g)
            v_pred = v_pred.squeeze(1)
            policy_loss = policy_cross_entropy(policy_logits, b_idx, b_prob,
                                               b_leg, b_leglen, version)
            value_loss = F.mse_loss(v_pred, v_target)
            loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            with torch.no_grad():
                tgt_ent = float(-(b_prob * b_prob.clamp_min(1e-12).log()).sum(dim=1).mean())
            print(f"step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"policy={policy_loss.item():.4f}（训练目标熵{tgt_ent:.4f}）  value={value_loss.item():.4f}")

        # 验证集评估（每 interval 步一次，纯前向，按局划分）
        if step > 0 and step % interval == 0 and v_boards.shape[0] > 0:
            model.eval()
            vps = vvs_ = 0.0
            vn = 0
            vent_sum = 0.0  # 验证集目标熵累计（与训练批的目标熵区分）
            vent_n = 0
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                for vstart in range(0, v_boards.shape[0], 256):
                    vx = torch.from_numpy(v_boards[vstart:vstart + 256]).to(device)
                    vg = torch.from_numpy(v_graves[vstart:vstart + 256]).to(device)
                    vy = torch.from_numpy(v_vals[vstart:vstart + 256]).to(device)
                    vpol, vval = model(vx, vg)
                    vval = vval.squeeze(1)
                    v_leg = (torch.from_numpy(v_leg_idx_np[vstart:vstart + 256]).to(device)
                             if v_leg_idx_np is not None else None)
                    v_leglen = (torch.from_numpy(v_leg_len_np[vstart:vstart + 256]).to(device)
                                if v_leg_len_np is not None else None)
                    vprob_b = torch.from_numpy(v_pol_prob_np[vstart:vstart + 256]).to(device)
                    vps += policy_cross_entropy(
                        vpol,
                        torch.from_numpy(v_pol_idx_np[vstart:vstart + 256]).to(device),
                        vprob_b,
                        v_leg, v_leglen, version).item()
                    vvs_ += F.mse_loss(vval.float(), vy).item()
                    # 验证集自己的目标分布熵（padding 位 prob=0 不产生贡献）
                    vent_sum += float(-(vprob_b * vprob_b.clamp_min(1e-12).log()).sum(dim=1).sum())
                    vent_n += vprob_b.size(0)
                    vn += 1
            v_pol, v_val = vps / max(1, vn), vvs_ / max(1, vn)
            vent = vent_sum / max(1, vent_n)
            model.train()
            print(f"[验证] step {step + 1}: policy={v_pol:.4f} value={v_val:.4f} "
                  f"目标熵(验证)={vent:.4f}", flush=True)

            # ── 需求 1/2：每个验证点保存一份带步数后缀的 .pt（命名不冲突）──
            ck_path = os.path.join(checkpoint_dir, f"{net_name}_step{step + 1}.pt")
            save_checkpoint(ck_path, model, optimizer, step + 1, config,
                            val_policy_loss=v_pol, val_value_loss=v_val)
            ckpts.append((step + 1, ck_path, v_pol, v_val))
            val_policy_hist.append((step + 1, v_pol))
            val_value_hist.append((step + 1, v_val))
            last_val_step = step + 1


    # ── 训练结束后的末次验证：让最终状态也参与对比 ──
    if (last_val_step is None or last_val_step != num_steps) \
            and v_boards.shape[0] > 0:
        model.eval()
        vps = vvs_ = 0.0
        vn = 0
        vent_sum = 0.0
        vent_n = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for vstart in range(0, v_boards.shape[0], 256):
                vx = torch.from_numpy(v_boards[vstart:vstart + 256]).to(device)
                vg = torch.from_numpy(v_graves[vstart:vstart + 256]).to(device)
                vy = torch.from_numpy(v_vals[vstart:vstart + 256]).to(device)
                vpol, vval = model(vx, vg)
                vval = vval.squeeze(1)
                v_leg = (torch.from_numpy(v_leg_idx_np[vstart:vstart + 256]).to(device)
                         if v_leg_idx_np is not None else None)
                v_leglen = (torch.from_numpy(v_leg_len_np[vstart:vstart + 256]).to(device)
                            if v_leg_len_np is not None else None)
                vprob_b = torch.from_numpy(v_pol_prob_np[vstart:vstart + 256]).to(device)
                vps += policy_cross_entropy(
                    vpol,
                    torch.from_numpy(v_pol_idx_np[vstart:vstart + 256]).to(device),
                    vprob_b,
                    v_leg, v_leglen, version).item()
                vvs_ += F.mse_loss(vval.float(), vy).item()
                vent_sum += float(-(vprob_b * vprob_b.clamp_min(1e-12).log()).sum(dim=1).sum())
                vent_n += vprob_b.size(0)
                vn += 1
        v_pol, v_val = vps / max(1, vn), vvs_ / max(1, vn)
        vent = vent_sum / max(1, vent_n)
        model.train()
        print(f"[末次验证] step {num_steps}: policy={v_pol:.4f} value={v_val:.4f} "
              f"目标熵(验证)={vent:.4f}", flush=True)
        ck_path = os.path.join(checkpoint_dir, f"{net_name}_step{num_steps}.pt")
        save_checkpoint(ck_path, model, optimizer, num_steps, config,
                        val_policy_loss=v_pol, val_value_loss=v_val)
        ckpts.append((num_steps, ck_path, v_pol, v_val))
        val_policy_hist.append((num_steps, v_pol))
        val_value_hist.append((num_steps, v_val))

    # ── 需求 3/5/6：训练后处理（最优改名 / 末版子文件夹 / ONNX 导出）──
    if ckpts:
        best = min(ckpts, key=lambda c: c[2])   # 全局 policy loss 最低
        last = max(ckpts, key=lambda c: c[0])   # 步数最大 = 最后版本

        # 2026-09-09（用户要求，替代原 value 早停）：
        # 主版本 = value loss < 1 的验证点中 policy loss 最低者 → 根目录【命名】.pt；
        # 全局 policy loss 最低但 value >= 1 的版本不能当主版本，
        # 作为副版本保存到 best_policy 子文件夹（与 last 副版本同方式）；
        # 若没有任何 value < 1 的验证点，主版本退化为最后版本（日志说明）。
        good = [c for c in ckpts if c[3] < 1.0]
        if good:
            main = min(good, key=lambda c: c[2])
            main_reason = f"value<1 的 {len(good)} 个验证点中 policy 最低"
        else:
            # 2026-09-10（用户澄清）：全程 value>=1 无合格候选时，
            # 主版本 = 全局 policy loss 最低者（value>=1 也产出主版本）
            main = best
            main_reason = "无 value<1 的验证点，主版本取全局 policy 最低（value>=1）"
        main_is_global_best = (main[0] == best[0])
        main_is_last = (main[0] == last[0])

        # 主版本改名为【命名】.pt（子文件夹外）
        final_pt = os.path.join(checkpoint_dir, f"{net_name}.pt")
        os.replace(main[1], final_pt)
        print(f"[后处理] 主版本: step {main[0]}（policy={main[2]:.4f}, value={main[3]:.4f}）"
              f"→ {os.path.basename(final_pt)}（{main_reason}）")

        # 副版本 1：最后版本（主版本 ≠ 最后时）→ last\【命名】_last.pt（规则不变）
        if not main_is_last:
            sub = os.path.join(checkpoint_dir, "last")
            os.makedirs(sub, exist_ok=True)
            dst = os.path.join(sub, f"{net_name}_last.pt")
            os.replace(last[1], dst)
            print(f"[后处理] 最后版本（step {last[0]}）→ {dst}")

        # 副版本 2：全局 policy 最低但 value >= 1 → best_policy\【命名】_best_policy.pt
        # （主版本即全局最低时不产生；与最后版本同一步数时已并入 last 副版本，不重复）
        if not main_is_global_best and best[0] != last[0]:
            sub2 = os.path.join(checkpoint_dir, "best_policy")
            os.makedirs(sub2, exist_ok=True)
            dst2 = os.path.join(sub2, f"{net_name}_best_policy.pt")
            os.replace(best[1], dst2)
            print(f"[后处理] policy 最低 {best[2]:.4f} 但 value={best[3]:.4f} >= 1（step {best[0]}）"
                  f"→ {dst2}（副版本）")

        # ONNX 导出源 = 主版本（根目录【命名】.pt，与面板【导出】按钮同源）
        onnx_ok = False
        try:
            from export_onnx import export as _export_onnx
            onnx_path = os.path.join(checkpoint_dir, f"{net_name}.onnx")
            _export_onnx(final_pt, onnx_path, config)
            onnx_ok = True
            print(f"[后处理] ONNX 导出（主版本 policy={main[2]:.4f}，step {main[0]}）→ {onnx_path}")
        except Exception as ex:
            print(f"[后处理] ONNX 导出失败: {ex}（保留全部中间 .pt 存档，可用面板【导出】按钮手动导出 ONNX）")
    else:
        # 无验证样本的防御路径：直接保存最终状态
        onnx_ok = False
        save_checkpoint(os.path.join(checkpoint_dir, f"{net_name}.pt"),
                        model, optimizer, num_steps, config)

    # 需求（用户 09-08）：清理中间存档——仅当 ONNX 导出成功；
    # 导出失败时保留全部 _step .pt，仍可手动导出 ONNX
    if onnx_ok:
        removed = 0
        for p in sorted(glob.glob(os.path.join(checkpoint_dir, f"{net_name}_step*.pt"))):
            os.remove(p)
            removed += 1
        if removed:
            print(f"[后处理] 已清理中间存档 {removed} 份（仅保留主版本与副版本）")

    # 需求 1：验证点 policy / value loss 折线图（保存在同一文件夹）
    def _plot_hist(hist, fname, ylabel, title, color):
        if not hist:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文标签可读
        plt.rcParams["axes.unicode_minus"] = False
        xs = [s for s, _ in hist]
        ys = [v for _, v in hist]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, marker="o", color=color)
        ax.set_xlabel("验证步数")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        path = os.path.join(checkpoint_dir, fname)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[后处理] 折线图 → {path}")

    try:
        _plot_hist(val_policy_hist, f"{net_name}_policy_loss.png",
                   "policy loss（验证集）", f"{net_name} 各验证点 policy loss", "#1a73e8")
        _plot_hist(val_value_hist, f"{net_name}_value_loss.png",
                   "value loss（验证集）", f"{net_name} 各验证点 value loss", "#e8710a")
    except Exception as ex:
        print(f"[后处理] 折线图生成失败: {ex}")
    print("训练完成")


def main():
    config = load_config("config.json")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints"
    resume_from = sys.argv[3] if len(sys.argv) > 3 else None
    net_name = sys.argv[4] if len(sys.argv) > 4 else "latest"

    boards, graves, pols, vals, v_boards, v_graves, v_pols, v_vals, version, leg_idx, leg_len, v_leg_idx, v_leg_len = load_data(data_dir)
    if boards is None:
        return
    train(config, (boards, graves, pols, vals, v_boards, v_graves, v_pols, v_vals,
                   version, leg_idx, leg_len, v_leg_idx, v_leg_len),
          checkpoint_dir, resume_from, net_name)


if __name__ == "__main__":
    main()
