# -*- coding: utf-8 -*-
"""训练操作面板：控制自对弈数据收集，实时观察进度。"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import sys
import subprocess
import time
import datetime
import threading

# ====================================================================
# 配置
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
COLLECTOR_DLL = os.path.join(BASE_DIR, "collector", "bin", "Release",
                            "net10.0", "SelfPlayCollector.dll")
RANDOM_INIT_PY = os.path.join(BASE_DIR, "random_init_net.py")


DOTNET = r"C:\Program Files\dotnet\dotnet.exe"
PYTHON_EXE = r"C:\Users\Lenovo\anaconda3\python.exe"
TRAIN_PY = os.path.join(BASE_DIR, "train.py")
EXPORT_ONNX_PY = os.path.join(BASE_DIR, "export_onnx.py")
MATCH_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "match_results")  # 对战结果存 training 之外
PLOT_MATCH_PY = os.path.join(BASE_DIR, "plot_match.py")


FONT = ("Microsoft YaHei", 10)
FONT_TITLE = ("Microsoft YaHei", 14, "bold")

# 超参数说明（路径 → (中文名, 说明, 是否结构类)）
PARAM_INFO = {
    "network.num_residual_blocks": ("残差块数", "网络层数，越深能学越复杂的棋型，但训练更慢、数据少时易过拟合", True),
    "network.channels": ("特征通道数", "每层卷积的特征通道数，越大网络越宽、表达力越强，但更慢更易过拟合", True),
    "training.learning_rate": ("学习率", "梯度更新步长，太大震荡、太小收敛慢", False),
    "training.batch_size": ("批大小", "每次梯度更新用多少样本，越大越稳但更吃显存", False),
    "training.weight_decay": ("权重衰减", "L2 正则化强度，防止网络死记训练数据（过拟合）", False),
    "training.num_train_steps": ("训练步数", "每一代训练多少步，步数越多学得越充分（也更慢）", False),
    "training.checkpoint_interval": ("存档间隔", "每隔多少步保存一次模型存档，中断后可恢复", False),
    "training.value_loss_weight": ("价值损失权重", "价值头损失占总损失的比重，越大越重视胜负判断", False),
        "mcts.lottery_eval_limit": ("抽奖候选评估上限", "搜索内每个新抽奖结果最多评估多少个候选效果（用子力启发式而非NN，避免评估风暴；升级/生成/复活类枚举可达数百上千）。推荐8~32", False),
        "mcts.virtual_loss": ("虚拟损失", "树内并行K个worker选路时先扣的临时失败分，用于互相避让。需大于真实回报尺度(终局±1)；过大会抑制探索(抽奖饿死)、过小避让不足。推荐0.3~1.0，当前0.5", False),
        "mcts.num_mcts_sims": ("MCTS 模拟次数", "自对弈与对战共用的每步模拟数（对战双方相同才公平，已统一由此参数控制）。常用150~300；越大越强但越慢", False),
        "mcts.eval_material_weight": ("评估子力权重", "两处用途：①纯MCTS的rollout未分胜负时按子力差给连续估值；②抽奖候选效果选择时的静态评估。输出压在±权重内。推荐0.1~0.2；设0=关闭", False),
    "mcts.cpuct": ("探索常数", "越大越爱尝试新走法（探索），越小越走当前最优（利用）", False),
    "mcts.temperature": ("温度", "走子随机程度：越高越随机，1=按概率采样，0=总选概率最高", False),
    "mcts.temp_threshold": ("温度阈值", "前多少步用温度随机采样，之后贪心选最优", False),
    "selfplay.dirichlet_alpha": ("Dirichlet α", "开局探索噪声强度，越大越鼓励尝试新走法", False),
    "selfplay.dirichlet_epsilon": ("Dirichlet ε", "噪声占先验概率的比例（0~1），越大越随机", False),
    "selfplay.max_moves": ("最大步数", "单局最大步数，超时判和", False),
        "selfplay.parallel_games": ("并行局数", "同时自对弈局数。神经模式下 局数×树内K≈在飞NN请求数，越大GPU批量越满；本机推荐24~48", False),
        "selfplay.mcts_threads": ("每局MCTS线程(树内并行K)", "每局搜索树内的并行worker数（虚拟损失共享树）。神经模式下NN请求密度×K；实测24局×4=96 worker可用。推荐2~8", False),
    "selfplay.neural_batch_size": ("神经网络批量大小", "GPU一次推理处理多少个局面。实测8GB显存训练batch128仅占1.1GB，批量推理侧很富余；推荐范围32~64，并发局多时可试128", False),
    "selfplay.neural_batch_timeout_ms": ("批量超时(ms)", "攒批最长等待毫秒数。低并发场景(对战≤8局、请求稀疏)推荐10~15让批次更满；高并发自对弈(24局+)用默认2~5延迟更低", False),
    "selfplay.match_parallel_games": ("对战并行局数", "同时进行多少局对战。每局同时只有一方在搜索(线程数=mcts_threads)，本机实测16局≈32线程可跑满；调大加快评测、调小减少CPU争抢", False),
    "training.use_amp": ("混合精度(fp16)", "训练用AMP：卷积等重算子在fp16加速，BN/softmax自动保持fp32，GradScaler防下溢。实测提速约1.5~2倍；若遇NaN先关此项排查", False),
}


PARAM_GROUPS = [
    ("网络结构（改动需从头训练）",
     ["network.num_residual_blocks", "network.channels"]),
    ("训练",
     ["training.learning_rate", "training.batch_size", "training.weight_decay",
      "training.num_train_steps", "training.checkpoint_interval",
      "training.value_loss_weight", "training.use_amp"]),
    ("全局搜索（训练与对战共用）",
     ["mcts.num_mcts_sims", "mcts.cpuct", "mcts.temperature",
      "mcts.temp_threshold", "mcts.eval_material_weight",
      "mcts.virtual_loss", "mcts.lottery_eval_limit"]),
    ("自对弈数据收集",
     ["selfplay.dirichlet_alpha", "selfplay.dirichlet_epsilon",
      "selfplay.max_moves", "selfplay.parallel_games",
      "selfplay.mcts_threads", "selfplay.neural_batch_size",
      "selfplay.neural_batch_timeout_ms"]),
    ("对战评测",
     ["selfplay.match_parallel_games"]),
]

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_nested(cfg, path):
    cur = cfg
    for key in path.split("."):
        cur = cur[key]
    return cur


def set_nested(cfg, path, value):
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


class ProgressBar:
    """Canvas 实现的进度条，文字透明背景覆盖在中间，不遮挡进度条。"""
    def __init__(self, parent, bar_color="#1a73e8"):
        self.canvas = tk.Canvas(parent, height=24, bg="white",
                                highlightthickness=0)
        self.canvas.pack(side="left", fill="x", expand=True, padx=5)
        self.bar_color = bar_color
        self.value = 0.0
        self.text = ""
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def set(self, value, text):
        self.value = value
        self.text = text
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        # 灰色底（trough）
        self.canvas.create_rectangle(0, 0, w, h, fill="#e0e0e0", outline="")
        # 蓝色填充
        fill_w = max(0, min(w, w * self.value / 100.0))
        if fill_w > 0:
            self.canvas.create_rectangle(0, 0, fill_w, h, fill=self.bar_color, outline="")
        # 透明文字（居中，覆盖在进度条上，无背景遮挡）
        self.canvas.create_text(w / 2, h / 2, text=self.text, fill="black",
                                font=("Microsoft YaHei", 10))


class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("训练操作面板")
        self.root.geometry("1000x680")
        self.root.resizable(True, True)

        # 状态机
        self.state = "ready"  # ready/training/paused/saving/finished
        self.process = None
        self.target_games = 0
        self.data_dir = None       # 本次训练的数据文件夹
        self.resume_folder = None  # 续训的源文件夹
        self.elapsed = 0.0         # 已累计秒数（不含暂停）
        self._timer_start = 0.0
        self._running = False      # 当前是否在训练段（计时用）
        self._last_done = 0        # 上次已完成的局数（日志检测用）
        self._last_perf_time = 0.0 # 上次性能采样的时间
        self._cpu_percent = 0.0     # CPU 占用率（后台线程读）
        self._mem_percent = 0.0     # 内存占用率
        self._gpu_percent = None    # GPU 占用率
        self.phase = "collect"      # collect=数据收集 / train=训练网络
        self.train_process = None   # train.py 子进程
        self._resume_pt = None      # 续训时加载的上一代 .pt 路径
        self._resume_onnx = None    # 上一代导出的 .onnx 路径（指导自对弈）
        self.generation = 0         # 当前代际（0=全新初始化，N=第 N 代迭代）

        # 对战状态
        self.match_state = "idle"   # idle/running/finished
        self.net1_path = None
        self.net2_path = None
        self.mcts2_overrides = {}
        self._match_pause_file = None
        self._match_paused = False
        self._match_t0 = 0.0
        self.match_process = None
        self.match_onnx = None
        self.match_dir = None
        self.match_target = 0
        self._match_log_path = None
        self._match_log_lines = 0

        self.build_ui()
        self.load_params_into_ui()
        self.log("操作面板已启动")
        # 后台线程定时采样 CPU/内存/GPU（避免阻塞 UI）
        threading.Thread(target=self._perf_monitor_loop, daemon=True).start()
        self.root.after(200, self.tick)
        self.root.mainloop()

    # ================================================================
    # UI 构建
    # ================================================================
    def build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        self.train_frame = ttk.Frame(nb)
        self.param_frame = ttk.Frame(nb)
        self.match_frame = ttk.Frame(nb)
        nb.add(self.train_frame, text="  训练界面  ")
        nb.add(self.param_frame, text="  超参数界面  ")
        nb.add(self.match_frame, text="  对战界面  ")

        self.build_train_ui()
        self.build_param_ui()
        self.build_match_ui()

    def build_train_ui(self):
        f = self.train_frame
        f.columnconfigure(0, weight=1)
        f.rowconfigure(7, weight=1)

        # 状态栏（白底无边框）
        self.status_label = tk.Label(f, text="训练未开始", font=FONT_TITLE,
                                     fg="#1a73e8", anchor="w", bg="white",
                                     relief="flat", bd=0)
        self.status_label.grid(row=0, column=0, sticky="we", padx=10, pady=(10, 2))

        # 计时器
        self.timer_label = tk.Label(f, text="00:00:00", font=("Consolas", 18),
                                    anchor="e", bg="white")
        self.timer_label.grid(row=0, column=1, sticky="e", padx=10)

        # 局数 + 网络名输入
        row = tk.Frame(f)
        row.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        tk.Label(row, text="训练局数：", font=FONT).pack(side="left")
        try:
            _default_games = str(load_config().get("selfplay", {}).get("num_games", 500))
        except Exception:
            _default_games = "500"
        self.games_var = tk.StringVar(value=_default_games)
        self.games_entry = tk.Entry(row, textvariable=self.games_var, width=10,
                                    font=FONT, justify="right")
        self.games_entry.pack(side="left", padx=5)
        tk.Label(row, text="网络名称：", font=FONT).pack(side="left", padx=(15, 2))
        self.net_name_var = tk.StringVar(value="latest")
        self.net_name_entry = tk.Entry(row, textvariable=self.net_name_var, width=16, font=FONT)
        self.net_name_entry.pack(side="left", padx=5)
        tk.Label(row, text="（保存为 名称.pt/.onnx）", font=FONT, fg="#666").pack(side="left")

        # 按钮
        btns = tk.Frame(f)
        btns.grid(row=2, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        self.start_btn = tk.Button(btns, text="开始", width=10, font=FONT,
                                   command=self.on_start, bg="#1a73e8", fg="white")
        self.start_btn.pack(side="left", padx=5)
        self.pause_btn = tk.Button(btns, text="暂停", width=10, font=FONT,
                                   command=self.on_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=5)
        self.stop_btn = tk.Button(btns, text="结束", width=10, font=FONT,
                                  command=self.on_stop)
        self.stop_btn.pack(side="left", padx=5)
        self.cancel_btn = tk.Button(btns, text="取消", width=10, font=FONT,
                                    command=self.on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=5)
        self.open_btn = tk.Button(btns, text="打开…", width=10, font=FONT,
                                  command=self.on_open)
        self.open_btn.pack(side="left", padx=5)
        self.view_btn = tk.Button(btns, text="查看", width=8, font=FONT,
                                  command=self.on_view)
        self.view_btn.pack(side="left", padx=5)

        # 进度条1：数据收集（按局数）
        p1 = tk.Frame(f)
        # 随机初始化网络（AlphaZero 式冷启动基线）：按钮 + 种子输入
        initrow = tk.Frame(f)
        initrow.grid(row=3, column=0, columnspan=2, sticky="we", padx=10, pady=(8, 2))
        self.init_btn = tk.Button(initrow, text="随机初始化网络", width=14, font=FONT,
                                  command=self.on_random_init, bg="#188038", fg="white")
        self.init_btn.pack(side="left", padx=5)
        tk.Label(initrow, text="输入种子：", font=FONT).pack(side="left", padx=(10, 2))
        self.init_seed_var = tk.StringVar()
        self.init_seed_entry = tk.Entry(initrow, textvariable=self.init_seed_var,
                                        width=12, font=FONT, justify="right")
        self.init_seed_entry.pack(side="left")
        tk.Label(initrow, text="（留空=系统随机；网络名用上方输入框）",
                 font=FONT, fg="#666").pack(side="left", padx=4)
        p1.grid(row=4, column=0, columnspan=2, sticky="we", padx=10, pady=(15, 2))
        tk.Label(p1, text="数据收集", font=FONT, width=8).pack(side="left")
        self.progress = ProgressBar(p1)

        # 进度条2：网络训练（按步数）
        p2 = tk.Frame(f)
        p2.grid(row=5, column=0, columnspan=2, sticky="we", padx=10, pady=2)
        tk.Label(p2, text="网络训练", font=FONT, width=8).pack(side="left")
        self.train_progress = ProgressBar(p2)

        # 性能监测（绿色进度条：CPU/GPU/内存）
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", background="#34a853",
                        troughcolor="#e0e0e0")
        perf = tk.LabelFrame(f, text=" 性能监测 ", font=FONT, padx=6, pady=4,
                             bg="white")
        perf.grid(row=6, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        perf.columnconfigure(1, weight=1)

        self.perf_bars = {}
        perf_rows = [("CPU", "cpu"), ("内存", "mem"), ("GPU", "gpu")]
        for r, (label, key) in enumerate(perf_rows):
            tk.Label(perf, text=label, font=FONT, width=6, anchor="e",
                     bg="white").grid(row=r, column=0, sticky="e", padx=4, pady=1)
            bar = ttk.Progressbar(perf, style="green.Horizontal.TProgressbar",
                                  orient="horizontal", maximum=100)
            bar.grid(row=r, column=1, sticky="we", padx=4, pady=1)
            val = tk.Label(perf, text="--", font=FONT, width=6, anchor="w",
                           bg="white")
            val.grid(row=r, column=2, sticky="w", padx=4, pady=1)
            self.perf_bars[key] = (bar, val)

        # 信息栏（黑框白底，可滚动，最多 100 条）
        info_frame = tk.Frame(f, relief="solid", bd=1, bg="white")
        info_frame.grid(row=7, column=0, columnspan=2, sticky="nsew",
                        padx=10, pady=(5, 10))
        self.info_text = tk.Text(info_frame, height=8, bg="white", fg="black",
                                 font=FONT, wrap="word", state="disabled")
        info_scroll = tk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scroll.pack(side="right", fill="y")

    def build_param_ui(self):
        f = self.param_frame
        self.param_canvas = tk.Canvas(f)
        canvas = self.param_canvas
        scroll = ttk.Scrollbar(f, orient="vertical", command=canvas.yview)
        self.param_inner = tk.Frame(canvas)
        self.param_inner.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # 表头
        tk.Label(self.param_inner, text="超参数", font=FONT, width=22,
                 anchor="w", bg="#eee").grid(row=0, column=0, sticky="we", padx=1, pady=1)
        tk.Label(self.param_inner, text="数值", font=FONT, width=12,
                 bg="#eee").grid(row=0, column=1, padx=1, pady=1)
        tk.Label(self.param_inner, text="说明", font=FONT, width=42, anchor="w",
                 bg="#eee").grid(row=0, column=2, sticky="we", padx=1, pady=1)

        self.param_entries = {}
        row_idx = 1
        for group_name, paths in PARAM_GROUPS:
            tk.Label(self.param_inner, text=group_name, font=FONT_TITLE,
                     fg="#174ea6", anchor="w", bg="#e8f0fe").grid(
                row=row_idx, column=0, columnspan=3, sticky="we", padx=1, pady=(6, 1))
            row_idx += 1
            for path in paths:
                name, desc, is_struct = PARAM_INFO[path]
                label = name + (" [结构]" if is_struct else "")
                tk.Label(self.param_inner, text=label, font=FONT, anchor="w",
                         fg="#c5221f" if is_struct else "black").grid(
                    row=row_idx, column=0, sticky="we", padx=1, pady=1)
                entry = tk.Entry(self.param_inner, width=12, font=FONT, justify="right")
                entry.grid(row=row_idx, column=1, padx=1, pady=1)
                self.param_entries[path] = entry
                tk.Label(self.param_inner, text=desc, font=FONT, anchor="w",
                         wraplength=420, justify="left").grid(
                    row=row_idx, column=2, sticky="we", padx=1, pady=1)
                row_idx += 1

        # 确认 / 取消
        btns = tk.Frame(self.param_inner)
        btns.grid(row=row_idx, column=0, columnspan=3, pady=15)
        tk.Button(btns, text="确认", width=12, font=FONT, bg="#1a73e8", fg="white",
                  command=self.on_confirm_params).pack(side="left", padx=10)
        tk.Button(btns, text="取消", width=12, font=FONT,
                  command=self.load_params_into_ui).pack(side="left", padx=10)

        # 鼠标滚轮滚动（canvas 和所有子 widget 都绑定）
        for w in [canvas] + list(self.param_inner.winfo_children()):
            w.bind("<MouseWheel>", self._on_param_wheel)

    def _on_param_wheel(self, event):
        self.param_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # ================================================================
    #  对战界面
    # ================================================================
    def build_match_ui(self):
        f = self.match_frame
        f.columnconfigure(0, weight=1)
        f.rowconfigure(5, weight=1)

        self.match_status_label = tk.Label(f, text="对战未开始", font=FONT_TITLE,
                                           fg="#1a73e8", anchor="w", bg="white")
        self.match_timer_label = tk.Label(f, text="00:00:00", font=("Consolas", 14),
                                          anchor="e", bg="white")
        self.match_status_label.grid(row=0, column=0, sticky="we", padx=10, pady=(10, 2))
        self.match_timer_label.grid(row=0, column=1, sticky="e", padx=10)

        row = tk.Frame(f)
        row.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        tk.Label(row, text="对局数：", font=FONT).pack(side="left")
        self.match_games_var = tk.StringVar(value="100")
        self.match_games_entry = tk.Entry(row, textvariable=self.match_games_var, width=10,
                                          font=FONT, justify="right")
        self.match_games_entry.pack(side="left", padx=5)
        tk.Label(row, text="文件夹名：", font=FONT).pack(side="left", padx=(10, 2))
        self.match_folder_var = tk.StringVar(value="")
        self.match_folder_entry = tk.Entry(row, textvariable=self.match_folder_var, width=14,
                                           font=FONT)
        self.match_folder_entry.pack(side="left", padx=5)
        tk.Label(row, text="（留空=用时间）", font=FONT, fg="#666").pack(side="left")
        self.match_prepare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="启用准备模式", font=FONT,
                       variable=self.match_prepare_var).pack(side="right", padx=10)

        btns = tk.Frame(f)
        btns.grid(row=2, column=0, sticky="we", padx=10, pady=5)
        self.net1_btn = tk.Button(btns, text="网络1", width=8, font=FONT,
                                  command=lambda: self.pick_net(1))
        self.net1_btn.pack(side="left", padx=4)
        self.net2_btn = tk.Button(btns, text="网络2", width=8, font=FONT,
                                  command=lambda: self.pick_net(2))
        self.net2_btn.pack(side="left", padx=4)
        self.match_start_btn = tk.Button(btns, text="开始", width=8, font=FONT,
                                         command=self.on_match_start, bg="#1a73e8", fg="white")
        self.match_start_btn.pack(side="left", padx=4)
        self.match_pause_btn = tk.Button(btns, text="暂停", width=8, font=FONT,
                                         command=self.toggle_match_pause, state="disabled")
        self.match_pause_btn.pack(side="left", padx=4)
        self.match_stop_btn = tk.Button(btns, text="结束", width=8, font=FONT,
                                        command=self.on_match_stop, state="disabled")
        self.match_stop_btn.pack(side="left", padx=4)
        self.match_cancel_btn = tk.Button(btns, text="取消", width=8, font=FONT,
                                          command=self.on_match_cancel, state="disabled")
        self.match_cancel_btn.pack(side="left", padx=4)
        self.mcts2_btn = tk.Button(btns, text="调节MCTS2参数", width=14, font=FONT,
                                   command=self.open_mcts2_dialog)
        self.mcts2_btn.pack(side="right", padx=4)

        self.match_onnx_label = tk.Label(f, text="当前选择：未选择（网络1 vs 纯MCTS）",
                                         font=FONT, fg="#666", anchor="w")
        self.match_onnx_label.grid(row=3, column=0, sticky="we", padx=10)

        p = tk.Frame(f)
        p.grid(row=4, column=0, sticky="we", padx=10, pady=(10, 2))
        tk.Label(p, text="对局进度", font=FONT, width=8).pack(side="left")
        self.match_progress = ProgressBar(p)

        info_frame = tk.Frame(f, relief="solid", bd=1, bg="white")
        info_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.match_info_text = tk.Text(info_frame, height=8, bg="white", fg="black",
                                       font=FONT, wrap="word", state="disabled")
        info_scroll = tk.Scrollbar(info_frame, command=self.match_info_text.yview)
        self.match_info_text.configure(yscrollcommand=info_scroll.set)
        self.match_info_text.pack(side="left", fill="both", expand=True)
        info_scroll.pack(side="right", fill="y")

    def match_log(self, text):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.match_info_text.config(state="normal")
        self.match_info_text.insert("end", line + "\n")
        lines = int(self.match_info_text.index("end-1c").split(".")[0])
        if lines > 200:
            self.match_info_text.delete("1.0", f"{lines - 200}.0")
        self.match_info_text.see("end")
        self.match_info_text.config(state="disabled")

    def pick_net(self, slot):
        path = filedialog.askopenfilename(
            title=f"选择网络{slot}的 onnx 文件",
            initialdir=RESULTS_DIR,
            filetypes=[("ONNX 模型", "*.onnx"), ("所有文件", "*.*")])
        if not path:
            return
        if slot == 1:
            self.net1_path = path
        else:
            self.net2_path = path
        self.update_net_label()

    def update_net_label(self):
        n1 = os.path.basename(self.net1_path) if self.net1_path else None
        n2 = os.path.basename(self.net2_path) if self.net2_path else None
        if n1 and n2:
            txt = f"网络1: {n1}  vs  网络2: {n2}"
        elif n1:
            txt = f"网络1: {n1}  vs  纯MCTS"
        elif n2:
            txt = f"纯MCTS  vs  网络2: {n2}"
        else:
            txt = "MCTS1 vs MCTS2（均未选择网络）"
        self.match_onnx_label.config(text="当前选择：" + txt)

    def open_mcts2_dialog(self):
        cfg = load_config()
        d = tk.Toplevel(self.root)
        d.title("调节MCTS2参数（仅影响第二个玩家）")
        d.geometry("380x300")
        d.grab_set()
        fields = [
            ("探索常数 C", "cpuct", "1.2"),
            ("模拟次数 sims", "num_mcts_sims", "200"),
            ("rollout深度 depth", "max_rollout_depth", "200"),
            ("评估子力权重 w", "eval_material_weight", "0.15"),
            ("虚拟损失 vl", "virtual_loss", "0.5"),
            ("抽奖倍率 mult", "lottery_multiplier", "1.0"),
        ]
        entries = {}
        for i, (label, key, dflt) in enumerate(fields):
            tk.Label(d, text=label, font=FONT).grid(row=i, column=0, sticky="w",
                                                    padx=10, pady=4)
            cur = cfg.get("mcts", {}).get(key, dflt)
            e = tk.Entry(d, width=12, font=FONT, justify="right")
            e.insert(0, str(cur))
            e.grid(row=i, column=1, padx=10, pady=4)
            entries[key] = e
        tk.Label(d, text="留空 = 继承全局值", font=FONT, fg="#666").grid(
            row=len(fields), column=0, columnspan=2)

        def confirm():
            ov = {}
            for key, e in entries.items():
                v = e.get().strip()
                if v:
                    ov[key] = v
            self.mcts2_overrides = ov
            spec = ",".join(f"{k}={v}" for k, v in ov.items()) or "-"
            self.mcts2_spec_var.set(spec)
            d.destroy()

        tk.Button(d, text="确定", width=10, font=FONT, bg="#1a73e8", fg="white",
                  command=confirm).grid(row=len(fields) + 1, column=0, columnspan=2,
                                        pady=10)

    def on_match_start(self):
        if self.match_state == "running":
            return
        try:
            self.match_target = int(self.match_games_var.get())
        except ValueError:
            messagebox.showerror("错误", "对局数必须是整数")
            return
        if self.match_target <= 0:
            messagebox.showerror("错误", "对局数必须 > 0")
            return

        folder_name = self.match_folder_var.get().strip()
        if not folder_name:
            folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.match_dir = os.path.join(MATCH_RESULTS_DIR, folder_name)
        os.makedirs(self.match_dir, exist_ok=True)
        progress_file = os.path.join(self.match_dir, "progress.txt")
        pause_file = os.path.join(self.match_dir, "pause.flag")
        if os.path.exists(pause_file):
            os.remove(pause_file)
        self._match_pause_file = pause_file
        self._match_paused = False

        prepare_flag = "1" if self.match_prepare_var.get() else "0"
        net1 = self.net1_path or "none"
        net2 = self.net2_path or "none"
        m2parts = [f"{k}={v}" for k, v in self.mcts2_overrides.items()]
        m2spec = ",".join(m2parts) if m2parts else "-"

        cmd = [DOTNET, COLLECTOR_DLL, "match", str(self.match_target), net1,
               self.match_dir, progress_file, pause_file, prepare_flag, net2, m2spec]
        log_path = os.path.join(self.match_dir, "log.txt")
        logf = open(log_path, "w", encoding="utf-8")
        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._match_log_path = log_path
        self._match_log_lines = 0
        try:
            self.match_process = subprocess.Popen(cmd, stdout=logf, stderr=logf,
                                                  creationflags=no_window)
        except Exception as e:
            messagebox.showerror("错误", f"无法启动对战进程：\n{e}")
            logf.close()
            return

        self.match_state = "running"
        self._match_paused = False
        self._match_t0 = time.time()
        self.match_progress.set(0, f"0/{self.match_target} (0.0%)")
        self.match_status_label.config(text="对战中", fg="#1a73e8")
        self.update_match_buttons()
        self.match_log(f"开始对战 × {self.match_target} 局")

    def toggle_match_pause(self):
        if self.match_state != "running":
            return
        pf = getattr(self, "_match_pause_file", None)
        if not pf:
            return
        if not self._match_paused:
            with open(pf, "w") as f:
                f.write("pause")
            self._match_paused = True
            self.match_pause_btn.config(text="继续")
            self.match_status_label.config(text="对战已暂停", fg="#e8710a")
            self.match_log("已暂停")
        else:
            if os.path.exists(pf):
                os.remove(pf)
            self._match_paused = False
            self.match_pause_btn.config(text="暂停")
            self.match_status_label.config(text="对战中", fg="#1a73e8")
            self.match_log("继续对战")

    def on_match_stop(self):
        if self.match_state != "running":
            return
        if self.match_process is not None and self.match_process.poll() is None:
            self.match_process.terminate()
            try:
                self.match_process.wait(timeout=5)
            except Exception:
                self.match_process.kill()
        self.match_state = "finished"
        self.match_status_label.config(text="对战已停止", fg="#e8710a")
        self.update_match_buttons()

    def on_match_cancel(self):
        if self.match_state not in ("running", "paused"):
            return
        ok = messagebox.askyesno("确认", "确定取消本次对战吗？\n（本次对战的所有记录将被删除）")
        if not ok:
            return
        if self.match_process is not None and self.match_process.poll() is None:
            self.match_process.terminate()
            try:
                self.match_process.wait(timeout=5)
            except Exception:
                self.match_process.kill()
        try:
            import shutil
            if self.match_dir and os.path.isdir(self.match_dir):
                shutil.rmtree(self.match_dir, ignore_errors=True)
        except Exception:
            pass
        self.match_state = "idle"
        self.match_process = None
        self.match_dir = None
        self._match_paused = False
        self.match_progress.set(0, "0/0 (0.0%)")
        self.match_status_label.config(text="对战未开始（已取消）", fg="#e8710a")
        self.update_match_buttons()
        self.match_log("已取消本次对战，记录已清除")

    def update_match_buttons(self):
        if self.match_state == "running":
            self.match_start_btn.config(state="disabled")
            self.net1_btn.config(state="disabled")
            self.net2_btn.config(state="disabled")
            self.mcts2_btn.config(state="disabled")
            self.match_games_entry.config(state="disabled")
            self.match_pause_btn.config(state="normal", text="暂停")
            self.match_stop_btn.config(state="normal")
            self.match_cancel_btn.config(state="normal")
        elif self.match_state == "paused":
            self.match_pause_btn.config(state="normal", text="继续")
            self.match_stop_btn.config(state="normal")
            self.match_cancel_btn.config(state="normal")
        else:
            self.match_start_btn.config(state="normal")
            self.net1_btn.config(state="normal")
            self.net2_btn.config(state="normal")
            self.mcts2_btn.config(state="normal")
            self.match_games_entry.config(state="normal")
            self.match_pause_btn.config(state="disabled", text="暂停")
            self.match_stop_btn.config(state="disabled")
            self.match_cancel_btn.config(state="disabled")

    def load_params_into_ui(self):
        cfg = load_config()
        for path, entry in self.param_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(get_nested(cfg, path)))

    def on_confirm_params(self):
        cfg = load_config()
        for path, entry in self.param_entries.items():
            try:
                raw = entry.get().strip()
                old = get_nested(cfg, path)
                if isinstance(old, bool):
                    val = raw.lower() in ("1", "true", "yes", "on", "开")
                elif isinstance(old, int):
                    val = int(float(raw))
                else:
                    val = float(raw)
                set_nested(cfg, path, val)
            except ValueError:
                messagebox.showerror("错误", f"超参数 {path} 的值不是数字")
                return
        save_config(cfg)
        messagebox.showinfo("已确认", "超参数已保存到 config.json")

    # ================================================================
    # 状态机
    # ================================================================
    def set_status(self, text, color="#1a73e8"):
        self.status_label.config(text=text, fg=color)

    def update_buttons(self):
        if self.state == "ready":
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled", text="暂停")
            self.stop_btn.config(state="normal", text="结束")
            self.cancel_btn.config(state="disabled")
            self.open_btn.config(state="normal")
            self.games_entry.config(state="normal")
        elif self.state == "training":
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal", text="暂停")
            self.stop_btn.config(state="normal", text="结束")
            self.cancel_btn.config(state="normal")
            self.open_btn.config(state="disabled")
            self.games_entry.config(state="disabled")
        elif self.state == "paused":
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal", text="继续")
            self.stop_btn.config(state="normal", text="结束")
            self.cancel_btn.config(state="normal")
            self.open_btn.config(state="disabled")
            self.games_entry.config(state="disabled")
        elif self.state == "saving":
            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.cancel_btn.config(state="disabled")
            self.open_btn.config(state="disabled")
        elif self.state == "finished":
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="normal", text="结束")
            self.cancel_btn.config(state="disabled")
            self.open_btn.config(state="normal")
            self.games_entry.config(state="normal")

    def on_start(self):
        if self.state not in ("ready", "finished"):
            return
        try:
            self.target_games = int(self.games_var.get())
        except ValueError:
            messagebox.showerror("错误", "训练局数必须是整数")
            return
        if self.target_games <= 0:
            messagebox.showerror("错误", "训练局数必须 > 0")
            return

        # 网络名称（允许用户输入带或不带 .pt/.onnx 后缀）
        self.net_name = self.normalize_network_name(self.net_name_var.get())
        self.net_name_var.set(self.net_name)

        # 创建数据文件夹（日期-时间）
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_dir = os.path.join(RESULTS_DIR, f"{ts}")
        os.makedirs(self.data_dir, exist_ok=True)
        data_sub = os.path.join(self.data_dir, "data")
        os.makedirs(data_sub, exist_ok=True)

        progress_file = os.path.join(self.data_dir, "progress.txt")
        pause_flag = os.path.join(BASE_DIR, "pause.flag")

        # 删除旧的暂停标志
        if os.path.exists(pause_flag):
            os.remove(pause_flag)

        # 立即更新进度条为 0/N（而不是等第一局完成）
        self.progress.set(0, f"0/{self.target_games} (0.0%)")

        # 网络训练进度条也立即置 0，点击"开始"就显示占比，而非等训练阶段
        try:
            total_steps = load_config()["training"]["num_train_steps"]
        except Exception:
            total_steps = "?"
        self.train_progress.set(0, f"0/{total_steps} (0.0%)")

        # 确定是否用上一代网络指导自对弈（AlphaZero 迭代闭环）
        onnx_path = None
        if self._resume_onnx and os.path.exists(self._resume_onnx):
            onnx_path = self._resume_onnx

        # 启动 collector 子进程
        cmd = [DOTNET, COLLECTOR_DLL, str(self.target_games), data_sub,
               progress_file, pause_flag]
        if onnx_path:
            cmd.append(onnx_path)
        log_path = os.path.join(self.data_dir, "log.txt")
        logf = open(log_path, "w", encoding="utf-8")
        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._log_path = log_path
        self._log_lines_read = 0
        try:
            self.process = subprocess.Popen(cmd, stdout=logf, stderr=logf,
                                            creationflags=no_window)
        except Exception as e:
            messagebox.showerror("错误", f"无法启动训练进程：\n{e}")
            logf.close()
            return

        self.state = "training"
        self.phase = "collect"
        self.train_process = None
        self.elapsed = 0.0
        self._timer_start = time.time()
        self._running = True
        self._last_done = 0
        self.set_status("收集数据中", "#1a73e8")
        if onnx_path:
            self.log(f"开始第 {self.generation} 代训练，目标 {self.target_games} 局（网络指导自对弈）")
        else:
            self.log(f"开始第 {self.generation} 代训练，目标 {self.target_games} 局（纯 MCTS 自对弈）")
        self.update_buttons()

    def on_random_init(self):
        r"""生成随机初始化网络（AlphaZero 式冷启动）：自定义名称输出到 results/<时间戳>_init/。"""
        seed_txt = self.init_seed_var.get().strip()
        if seed_txt:
            try:
                seed = int(seed_txt)
            except ValueError:
                messagebox.showerror("错误", "初始化种子必须是整数")
                return
        else:
            seed = int(time.time() * 1000) % (2 ** 31)
            self.init_seed_var.set(str(seed))
        net_name = self.normalize_network_name(self.net_name_var.get())
        if not net_name:
            messagebox.showerror("错误", "请先填写网络名称")
            return
        out_dir = os.path.join(RESULTS_DIR, f"{datetime.datetime.now():%Y%m%d-%H%M%S}_init")
        os.makedirs(out_dir, exist_ok=True)
        self.set_status("随机初始化网络中…", "#188038")
        self.log(f"随机初始化网络：种子={seed} 网络名={net_name} 输出={out_dir}")
        self.init_btn.config(state="disabled")

        def worker():
            try:
                no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                pr = subprocess.run([PYTHON_EXE, RANDOM_INIT_PY, str(seed), out_dir, net_name],
                                    cwd=BASE_DIR, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=300,
                                    creationflags=no_window)
                out = (pr.stdout or "")
                if pr.returncode != 0 and pr.stderr:
                    out += "\n" + pr.stderr
                for line in out.splitlines():
                    self.after(0, lambda l=line: self.log(l))
                ok = pr.returncode == 0 and os.path.exists(os.path.join(out_dir, net_name + ".onnx"))
                def done():
                    self.log("随机初始化完成，.pt/.onnx/init_info.txt 已输出" if ok
                             else "随机初始化失败，详见上方输出")
                    self.set_status("随机初始化完成" if ok else "随机初始化失败",
                                    "#188038" if ok else "#c5221f")
                self.after(0, done)
            except Exception as e:
                self.after(0, lambda: self.log(f"随机初始化异常: {e}"))
            finally:
                self.after(0, lambda: self.init_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def on_pause(self):
        if self.state == "training":
            # 暂停：写 pause.flag
            pause_flag = os.path.join(BASE_DIR, "pause.flag")
            with open(pause_flag, "w") as f:
                f.write("pause")
            self.state = "paused"
            self.elapsed += time.time() - self._timer_start
            self._running = False
            self.set_status("训练已暂停", "#e8710a")
            self.update_buttons()
            self.log("训练已暂停")
        elif self.state == "paused":
            # 继续：删 pause.flag
            pause_flag = os.path.join(BASE_DIR, "pause.flag")
            if os.path.exists(pause_flag):
                os.remove(pause_flag)
            self.state = "training"
            self._timer_start = time.time()
            self._running = True
            self.set_status("训练中", "#1a73e8")
            self.update_buttons()
            self.log("训练继续")

    def on_stop(self):
        if self.state == "ready":
            self.root.destroy()
            return
        if self.state == "finished":
            self.root.destroy()
            return
        if self.state in ("training", "paused"):
            prev_state = self.state       # 记住之前是训练中还是暂停中
            prev_running = self._running
            # 先暂停 + 保存
            self.state = "saving"
            self.set_status("保存中", "#e8710a")
            self.update_buttons()
            # 二次确认
            ok = messagebox.askyesno("确认", "是否提前结束训练？\n（训练结果会先保存）")
            if not ok:
                # 否：恢复之前的训练/暂停状态（无论进程是否还在）
                self.state = prev_state
                if prev_state == "training":
                    pause_flag = os.path.join(BASE_DIR, "pause.flag")
                    if os.path.exists(pause_flag):
                        os.remove(pause_flag)
                    self._timer_start = time.time()
                    self._running = True
                    self.set_status("训练中", "#1a73e8")
                else:
                    self._running = prev_running
                    self.set_status("训练已暂停", "#e8710a")
                self.update_buttons()
                return
            # 是：终止进程 + 保存
            self.finish_training(interrupted=True)
            self.root.destroy()

    def on_cancel(self):
        """取消训练：停止进程、删除本次所有数据、回到可重新开始状态（不关程序）。"""
        if self.state not in ("training", "paused"):
            return
        ok = messagebox.askyesno("确认", "确定取消本次训练吗？\n（本次训练的所有数据将被删除，不保存）")
        if not ok:
            return
        pause_flag = os.path.join(BASE_DIR, "pause.flag")
        try:
            if self.phase == "collect" and self.process is not None and self.process.poll() is None:
                with open(pause_flag, "w") as f:
                    f.write("pause")
                time.sleep(0.3)
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except Exception:
                    self.process.kill()
            elif self.phase == "train" and self.train_process is not None and self.train_process.poll() is None:
                self.train_process.terminate()
                try:
                    self.train_process.wait(timeout=5)
                except Exception:
                    self.train_process.kill()
        except Exception:
            pass
        if os.path.exists(pause_flag):
            try:
                os.remove(pause_flag)
            except Exception:
                pass
        try:
            import shutil
            if self.data_dir and os.path.isdir(self.data_dir):
                shutil.rmtree(self.data_dir, ignore_errors=True)
        except Exception:
            pass
        self.state = "ready"
        self.process = None
        self.train_process = None
        self.data_dir = None
        self.phase = "collect"
        self._running = False
        self.elapsed = 0.0
        self.progress.set(0, "0/0 (0.0%)")
        try:
            total_steps = load_config()["training"]["num_train_steps"]
        except Exception:
            total_steps = "?"
        self.train_progress.set(0, f"0/{total_steps} (0.0%)")
        self.set_status("训练未开始（已取消）", "#e8710a")
        self.update_buttons()
        self.log("已取消本次训练，记录已清除")

    def finish_training(self, interrupted=False):
        """终止当前阶段的子进程，保存结果，进入完成状态（状态切换保证一定执行）。"""
        pause_flag = os.path.join(BASE_DIR, "pause.flag")
        # 终止当前阶段的进程（try/except，避免异常中断状态切换）
        try:
            if self.phase == "collect" and self.process is not None and self.process.poll() is None:
                with open(pause_flag, "w") as f:
                    f.write("pause")
                time.sleep(0.3)
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except Exception:
                    self.process.kill()
            elif self.phase == "train" and self.train_process is not None and self.train_process.poll() is None:
                self.train_process.terminate()
                try:
                    self.train_process.wait(timeout=5)
                except Exception:
                    self.train_process.kill()
        except Exception:
            pass
        if os.path.exists(pause_flag):
            try:
                os.remove(pause_flag)
            except Exception:
                pass

        # 保存结果（try/except）
        try:
            if self.data_dir:
                self.save_result_txt(self.data_dir, interrupted)
        except Exception:
            pass

        # 训练完成后清理临时文件 + 进度条置满（try/except，避免文件占用导致异常）
        if not interrupted and self.phase == "train" and self.data_dir:
            try:
                import shutil
                data_dir = os.path.join(self.data_dir, "data")
                if os.path.isdir(data_dir):
                    shutil.rmtree(data_dir)
                # 删除过程文件 progress.txt（纯进度通信，结束即无用）
                progress_file = os.path.join(self.data_dir, "progress.txt")
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                self.log("已自动删除训练数据 .bin 和进度文件（网络已保存，节省磁盘空间）")
            except Exception:
                pass
            # 进度条置满 100%
            try:
                total_steps = load_config()["training"]["num_train_steps"]
                self.train_progress.set(100, f"{total_steps}/{total_steps} (100%)")
            except Exception:
                self.train_progress.set(100, "完成 (100%)")

            # 导出 onnx（供下一代自对弈用网络指导），并记录 resume 指针
            net_name = self.normalize_network_name(getattr(self, "net_name", "latest"))
            pt_path = os.path.join(self.data_dir, f"{net_name}.pt")
            if os.path.exists(pt_path):
                onnx_out = os.path.join(self.data_dir, f"{net_name}.onnx")
                self.log("导出 ONNX 模型供下一代自对弈…")
                export_ok = self._export_onnx(pt_path, onnx_out)
                self._resume_pt = pt_path
                if export_ok and os.path.exists(onnx_out):
                    self._resume_onnx = onnx_out
                    self.log("ONNX 导出成功，下一代将用网络指导自对弈")
                else:
                    self._resume_onnx = None
                    self.log(f"ONNX 导出失败：未生成 {os.path.basename(onnx_out)}，下一代将退回纯 MCTS")
            else:
                self.log(f"找不到训练输出 {os.path.basename(pt_path)}，跳过 ONNX 导出")
            self.generation += 1

        # 状态切换（一定执行）
        if self._running:
            self.elapsed += time.time() - self._timer_start
            self._running = False
        self.state = "finished"
        if interrupted:
            self.set_status("训练中断", "#e8710a")
        elif self.generation - 1 == 0:
            self.set_status("初始化完成", "#188038")
        else:
            self.set_status("更新完成", "#188038")
        self.update_buttons()
        total_sec = int(self.elapsed)
        if interrupted:
            self.log(f"训练中断，用时 {total_sec} 秒，阶段={self.phase}")
        else:
            kind = "初始化" if self.generation - 1 == 0 else "更新"
            self.log(f"{kind}完成，用时 {total_sec} 秒，网络文件已保存")

    @staticmethod
    def normalize_network_name(name):
        """统一用户输入的网络文件名，避免生成 latest.pt 或名称.pt.pt。"""
        name = (name or "").strip()
        for suffix in (".onnx", ".pt"):
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)].rstrip()
        return name or "latest"

    def _export_onnx(self, pt_path, onnx_path):
        """调用 export_onnx.py 把 .pt 导出成 .onnx（供网络指导自对弈）。"""
        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            r = subprocess.run(
                [PYTHON_EXE, EXPORT_ONNX_PY, pt_path, onnx_path],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=600, creationflags=no_window,
                cwd=BASE_DIR)
            output = "\n".join(part for part in (r.stdout, r.stderr) if part).strip()
            if output:
                self.log(f"ONNX 导出日志：{output[-1000:]}")
            if r.returncode != 0:
                self.log(f"ONNX 导出进程失败（退出码 {r.returncode}）")
                return False
            return os.path.exists(onnx_path)
        except Exception as e:
            self.log(f"ONNX 导出异常: {e}")
            return False

    def save_result_txt(self, folder, interrupted):
        cfg = load_config()
        # 给人看的 txt
        with open(os.path.join(folder, "训练记录.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 40 + "\n")
            f.write(f"训练结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"状态: {'中断' if interrupted else '完成'}\n")
            f.write(f"训练局数: {self.target_games}\n")
            f.write("\n--- 超参数 ---\n")
            for group, params in cfg.items():
                f.write(f"\n[{group}]\n")
                for k, v in params.items():
                    f.write(f"  {k}: {v}\n")
        # 给程序解析的 JSON 快照（续训检测用）
        snapshot = {
            "训练结束时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "状态": "中断" if interrupted else "完成",
            "训练局数": self.target_games,
            "超参数": cfg,
        }
        with open(os.path.join(folder, "config_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    def on_open(self):
        """续训：选择结果文件夹，检测超参数一致性 + 检查网络文件。"""
        folder = filedialog.askdirectory(title="选择训练结果文件夹",
                                         initialdir=RESULTS_DIR)
        if not folder:
            return
        snapshot_path = os.path.join(folder, "config_snapshot.json")
        if not os.path.exists(snapshot_path):
            messagebox.showerror("错误", "所选文件夹里没有 config_snapshot.json\n（可能不是训练结果文件夹）")
            return
        # 检查网络文件（.pt）是否存在（兼容自定义名称，取第一个 .pt）
        import glob as _glob
        _pt_files = _glob.glob(os.path.join(folder, "*.pt"))
        if not _pt_files:
            messagebox.showerror("错误", "所选结果里没有网络文件 .pt\n（该结果未训练出网络，无法续训）")
            return
        self._resume_pt = _pt_files[0]
        # 网络名 = .pt 文件名（不含扩展名）
        self.net_name = self.normalize_network_name(
            os.path.splitext(os.path.basename(self._resume_pt))[0])
        self.net_name_var.set(self.net_name)

        # 读快照里的超参数
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        recorded_cfg = snapshot.get("超参数", {})

        # 检测结构类超参数一致性
        cfg = load_config()
        mismatches = []
        for path, (name, _, is_struct) in PARAM_INFO.items():
            if not is_struct:
                continue
            try:
                cur = get_nested(cfg, path)
                rec = get_nested(recorded_cfg, path)
            except (KeyError, TypeError):
                continue
            if rec != cur:
                mismatches.append((name, rec, cur))

        if mismatches:
            self.state = "ready"
            self.set_status("超参数不一致，无法开始", "#c5221f")
            self.start_btn.config(state="disabled")
            self.log("结构超参数不一致（改了这些必须从头训练）：")
            for n, r, c in mismatches:
                self.log(f"  {n}: 记录={r}, 当前={c}")
            messagebox.showerror(
                "无法续训",
                "结构超参数不一致，必须从头训练：\n" +
                "\n".join(f"  {n}: 记录={r}, 当前={c}" for n, r, c in mismatches))
            self.update_buttons()
            return

        # 一致：记录续训源文件夹
        self.resume_folder = folder
        self.generation = 1  # 已有网络，导入续训视为第 1 代（网络指导迭代）
        # 导出/复用 onnx（供网络指导自对弈）
        self._resume_onnx = os.path.join(folder, f"{self.net_name}.onnx")
        if not os.path.exists(self._resume_onnx):
            self.log("导出 ONNX 模型供网络指导自对弈…")
            self._export_onnx(self._resume_pt, self._resume_onnx)
            if os.path.exists(self._resume_onnx):
                self.log("ONNX 导出成功")
            else:
                self.log("ONNX 导出失败，将退回纯 MCTS 自对弈")
                self._resume_onnx = None
        self.set_status("训练未开始（已载入续训记录）", "#1a73e8")
        self.log(f"续训源: {folder}")
        self.log("将新建文件夹保存结果，不覆盖原记录")
        self.start_btn.config(state="normal")
        self.update_buttons()

    def on_view(self):
        """单击打开 training 文件夹。"""
        try:
            os.startfile(BASE_DIR)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件夹：\n{e}")

    # ================================================================
    # 日志 & 性能监测
    # ================================================================
    def log(self, text):
        """信息栏追加一条日志（带时间戳），最多保留 100 条。"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.info_text.config(state="normal")
        self.info_text.insert("end", line + "\n")
        lines = int(self.info_text.index("end-1c").split(".")[0])
        if lines > 100:
            self.info_text.delete("1.0", f"{lines - 100}.0")
        self.info_text.see("end")
        self.info_text.config(state="disabled")

    def read_gpu_percent(self):
        """同步读 GPU（后台线程专用，避免阻塞 UI）。"""
        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            out = subprocess.run(
                [r"C:\Windows\System32\nvidia-smi.exe",
                 "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, creationflags=no_window)
            return float(out.stdout.strip())
        except Exception:
            return None

    def _perf_monitor_loop(self):
        """后台线程：持续采样 CPU/内存/GPU，结果存实例变量。"""
        try:
            import psutil
            psutil.cpu_percent(interval=None)  # 初始化基线
        except Exception:
            return
        while True:
            try:
                self._cpu_percent = psutil.cpu_percent(interval=1.0)  # 阻塞 1 秒真实采样
                self._mem_percent = psutil.virtual_memory().percent
                self._gpu_percent = self.read_gpu_percent()
            except Exception:
                pass
            time.sleep(0.5)

    def update_performance(self):
        try:
            bar, val = self.perf_bars["cpu"]
            bar["value"] = self._cpu_percent
            val.config(text=f"{self._cpu_percent:.0f}%")
            bar, val = self.perf_bars["mem"]
            bar["value"] = self._mem_percent
            val.config(text=f"{self._mem_percent:.0f}%")
            if self._gpu_percent is not None:
                bar, val = self.perf_bars["gpu"]
                bar["value"] = self._gpu_percent
                val.config(text=f"{self._gpu_percent:.0f}%")
        except Exception:
            pass

    # ================================================================
    # 定时器
    # ================================================================
    def tick(self):
        # 计时
        if self._running:
            total = self.elapsed + (time.time() - self._timer_start)
        else:
            total = self.elapsed
        self.timer_label.config(text=time.strftime("%H:%M:%S", time.gmtime(total)))

        # 数据收集进度（从 progress.txt，按局数）
        if self.state in ("training", "paused", "saving") and self.data_dir:
            progress_file = os.path.join(self.data_dir, "progress.txt")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, "r") as f:
                        content = f.read().strip()
                    done, total = content.split("/")
                    done, total = int(done), int(total)
                    pct = (done / total * 100) if total > 0 else 0
                    self.progress.set(pct, f"{done}/{total} ({pct:.1f}%)")
                except Exception:
                    pass

        # 网络训练进度（从 train_log.txt 解析 step，按训练步数）
        if self.state == "training" and self.phase == "train":
            train_log = getattr(self, "_train_log_path", None)
            if train_log and os.path.exists(train_log):
                try:
                    import re
                    with open(train_log, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    matches = re.findall(r"step (\d+)/(\d+)", content)
                    if matches:
                        step, total = map(int, matches[-1])
                        pct = (step / total * 100) if total > 0 else 0
                        self.train_progress.set(pct, f"{step}/{total} ({pct:.1f}%)")
                except Exception:
                    pass

        # 读当前阶段的日志追加到信息栏
        if self.state in ("training", "paused", "saving"):
            log_path = None
            if self.phase == "collect":
                log_path = getattr(self, "_log_path", None)
                lines_read = getattr(self, "_log_lines_read", 0)
            else:
                log_path = getattr(self, "_train_log_path", None)
                lines_read = getattr(self, "_train_log_lines_read", 0)
            if log_path:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    if len(lines) > lines_read:
                        for line in lines[lines_read:]:
                            line = line.strip()
                            if line:
                                self.log(line)
                        lines_read = len(lines)
                        if self.phase == "collect":
                            self._log_lines_read = lines_read
                        else:
                            self._train_log_lines_read = lines_read
                except Exception:
                    pass

        # 性能监测（每 2 秒采样一次，降低开销）
        if time.time() - self._last_perf_time >= 2:
            self._last_perf_time = time.time()
            self.update_performance()

        # 阶段切换检测（数据收集完成 → 自动启动网络训练）
        if self.state == "training":
            if self.phase == "collect" and self.process is not None:
                if self.process.poll() is not None:
                    if self.process.returncode != 0:
                        self.finish_training(interrupted=True)
                    else:
                        self.start_train_phase()
            elif self.phase == "train" and self.train_process is not None:
                if self.train_process.poll() is not None:
                    self.finish_training(interrupted=(self.train_process.returncode != 0))

        # 对战进度（从 progress.txt 按局数）
        # 对战计时器
        if self.match_state == "running" and not getattr(self, "_match_paused", False):
            self.match_timer_label.config(text=time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - self._match_t0)))

        if self.match_state == "running" and self.match_dir:
            m_progress = os.path.join(self.match_dir, "progress.txt")
            if os.path.exists(m_progress):
                try:
                    with open(m_progress, "r") as f:
                        content = f.read().strip()
                    done, total = content.split("/")
                    done, total = int(done), int(total)
                    pct = (done / total * 100) if total > 0 else 0
                    self.match_progress.set(pct, f"{done}/{total} ({pct:.1f}%)")
                except Exception:
                    pass

            # 读对战日志
            if self._match_log_path and os.path.exists(self._match_log_path):
                try:
                    with open(self._match_log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    if len(lines) > self._match_log_lines:
                        for line in lines[self._match_log_lines:]:
                            line = line.strip()
                            if line:
                                self.match_log(line)
                        self._match_log_lines = len(lines)
                except Exception:
                    pass

            # 检测对战完成
            if self.match_process is not None and self.match_process.poll() is not None:
                self.match_state = "finished"
                self.match_status_label.config(text="对战完成", fg="#188038")
                self.update_match_buttons()
                summary_path = os.path.join(self.match_dir, "汇总.txt")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    self.match_log(line)
                    except Exception:
                        pass
                self.match_log(f"结果已保存: {self.match_dir}")
                # 删除 progress.txt（纯进度通信，结束即无用）
                try:
                    _pf = os.path.join(self.match_dir, "progress.txt")
                    if os.path.exists(_pf):
                        os.remove(_pf)
                except Exception:
                    pass
                # 自动生成三张图表
                _folder_name = os.path.basename(self.match_dir)
                csv_path = os.path.join(self.match_dir, f"{_folder_name}对局记录.csv")
                if os.path.exists(csv_path):
                    try:
                        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                        if self.net1_path is None and self.net2_path is None:
                            self.match_log("MCTS vs MCTS 模式不生成统计图")
                        else:
                            n1 = (os.path.splitext(os.path.basename(self.net1_path))[0]
                                  if self.net1_path else "纯MCTS")
                            n2 = (os.path.splitext(os.path.basename(self.net2_path))[0]
                                  if self.net2_path else "纯MCTS")
                            if self.net1_path is None and self.net2_path is None:
                                self.match_log("MCTS vs MCTS 模式不生成统计图")
                            else:
                                try:
                                    subprocess.run([PYTHON_EXE, PLOT_MATCH_PY, csv_path,
                                                    n1, n2],
                                                   creationflags=no_window, timeout=60)
                                except Exception as e:
                                    self.match_log(f"绘图失败: {e}")
                            self.match_log("已自动生成 3 张图表")
                    except Exception as e:
                        self.match_log(f"生成图表失败: {e}")

        self.root.after(200, self.tick)

    def start_train_phase(self):
        """数据收集完成，自动启动 train.py 训练网络（.pt 存根目录，与 txt/json 并列）。"""
        self.phase = "train"
        data_dir = os.path.join(self.data_dir, "data")
        checkpoint_dir = self.data_dir  # .pt 直接放结果文件夹根目录
        train_log = os.path.join(self.data_dir, "train_log.txt")
        logf = open(train_log, "w", encoding="utf-8")
        no_window = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        cmd = [PYTHON_EXE, TRAIN_PY, data_dir, checkpoint_dir]
        # 续训：加载上一代的 .pt 继续训练
        if getattr(self, "_resume_pt", None) and os.path.exists(self._resume_pt):
            cmd.append(self._resume_pt)
            self.log(f"续训：从 {os.path.basename(self._resume_pt)} 加载网络继续训练")
        else:
            # train.py 的第 3 个位置参数是 resume_from，需保留空位给网络名称
            cmd.append("")
            self.log("阶段2：网络初始化训练（数据收集完成）")
        # 网络保存名称（第 4 个参数）
        cmd.append(getattr(self, "net_name", "latest") or "latest")
        logf.write(f"[panel] 网络名称: {self.normalize_network_name(getattr(self, 'net_name', 'latest'))}\n")
        logf.flush()
        self.train_process = subprocess.Popen(cmd, stdout=logf, stderr=logf,
                                              creationflags=no_window)
        self._train_log_path = train_log
        self._train_log_lines_read = 0
        if getattr(self, "_resume_pt", None) and os.path.exists(self._resume_pt):
            self.set_status("更新网络中", "#1a73e8")
        else:
            self.set_status("初始化网络中", "#1a73e8")
        try:
            cfg = load_config()
            total_steps = cfg["training"]["num_train_steps"]
        except Exception:
            total_steps = "?"
        self.train_progress.set(0, f"0/{total_steps} (0%)")


if __name__ == "__main__":
    Panel()
