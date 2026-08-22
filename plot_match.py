# -*- coding: utf-8 -*-
"""对战结果可视化：读取 对局记录.csv，生成三张图。

用法: python plot_match.py <对局记录.csv>
输出到 csv 同目录下三张 PNG：
  图1_抽奖次数分布.png     —— onnx 抽奖次数区间柱状图，颜色区分胜负
  图2_抽奖占比散点.png     —— onnx 每局抽奖占比散点图，颜色区分胜负
  图3_双方抽奖散点.png     —— 双方抽奖数 vs 总步数散点，颜色=是否有网络，形状=胜负
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，供脚本/面板调用
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

WIN_COLOR = "#2e7d32"   # 绿 = onnx 胜
LOSE_COLOR = "#c62828"  # 红 = onnx 负
DRAW_COLOR = "#9e9e9e"  # 灰 = 和
ONNX_COLOR = "#1a73e8"  # 蓝 = 网络
MCTS_COLOR = "#e8710a"  # 橙 = 纯 MCTS


def load_records(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "game": int(r["局号"]),
                "result": r["结果"],
                "moves": int(r["步数"]),
                "onnx_lottery": int(r["onnx抽奖次数"]),
                "mcts_lottery": int(r["纯MCTS抽奖次数"]),
            })
    return rows


def plot1(rows, out):
    """图1：onnx 抽奖次数区间柱状图，颜色区分胜负，柱顶标数值，表头带胜率。"""
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    bounds = [(0, 1), (1, 3), (3, 6), (6, 11), (11, 21), (21, 10 ** 9)]
    win = [0] * len(labels)
    lose = [0] * len(labels)
    draw = [0] * len(labels)
    for r in rows:
        n = r["onnx_lottery"]
        idx = next(i for i, (lo, hi) in enumerate(bounds) if lo <= n < hi)
        if r["result"] == "onnx胜":
            win[idx] += 1
        elif r["result"] == "onnx负":
            lose[idx] += 1
        else:
            draw[idx] += 1

    total = len(rows)
    win_count = sum(1 for r in rows if r["result"] == "onnx胜")
    win_rate = win_count / total * 100 if total else 0

    x = np.arange(len(labels))
    w = 0.28
    plt.figure(figsize=(9, 5.5))
    bars1 = plt.bar(x - w, win, w, label="onnx胜", color=WIN_COLOR)
    bars2 = plt.bar(x, lose, w, label="onnx负", color=LOSE_COLOR)
    bars3 = plt.bar(x + w, draw, w, label="和", color=DRAW_COLOR)
    for bars in (bars1, bars2, bars3):
        plt.bar_label(bars, padding=2, fontsize=8)
    plt.xlabel("onnx 抽奖次数区间")
    plt.ylabel("局数")
    plt.title(f"图1：onnx 抽奖次数分布（按胜负分色）  总胜率 {win_rate:.1f}%")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot2(rows, out):
    """图2：onnx 每局抽奖占比散点图，颜色区分胜负。"""
    plt.figure(figsize=(8, 5))
    for r in rows:
        ratio = r["onnx_lottery"] / max(1, r["moves"]) * 100.0
        color = WIN_COLOR if r["result"] == "onnx胜" else (LOSE_COLOR if r["result"] == "onnx负" else DRAW_COLOR)
        plt.scatter(r["game"], ratio, c=color, s=20, alpha=0.7)
    handles = [mpatches.Patch(color=c, label=l) for c, l in
               [(WIN_COLOR, "onnx胜"), (LOSE_COLOR, "onnx负"), (DRAW_COLOR, "和")]]
    plt.legend(handles=handles)
    plt.xlabel("局号")
    plt.ylabel("onnx 抽奖占比（抽奖次数 / 步数，%）")
    plt.title("图2：onnx 每局抽奖占比（按胜负分色）")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot3(rows, out):
    """图3：双方抽奖数 vs 总步数散点，颜色=是否有网络，形状=胜负。"""
    plt.figure(figsize=(8, 5))
    markers = {"onnx胜": "o", "onnx负": "x", "和": "s"}
    for r in rows:
        mk = markers[r["result"]]
        plt.scatter(r["moves"], r["onnx_lottery"], c=ONNX_COLOR, marker=mk, s=22, alpha=0.6)
        plt.scatter(r["moves"], r["mcts_lottery"], c=MCTS_COLOR, marker=mk, s=22, alpha=0.6)
    handles = [
        mpatches.Patch(color=ONNX_COLOR, label="onnx（网络）"),
        mpatches.Patch(color=MCTS_COLOR, label="纯 MCTS"),
    ]
    mhandles = [plt.Line2D([], [], color="black", marker=m, linestyle="", label=l)
                for m, l in zip(["o", "x", "s"], ["onnx胜", "onnx负", "和"])]
    plt.legend(handles=handles + mhandles)
    plt.xlabel("总步数")
    plt.ylabel("抽奖次数")
    plt.title("图3：双方抽奖数 vs 总步数（颜色=是否有网络，形状=胜负）")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main(csv_path):
    outdir = os.path.dirname(os.path.abspath(csv_path))
    rows = load_records(csv_path)
    if not rows:
        print("[错误] 对局记录为空")
        return
    plot1(rows, os.path.join(outdir, "图1_抽奖次数分布.png"))
    plot2(rows, os.path.join(outdir, "图2_抽奖占比散点.png"))
    plot3(rows, os.path.join(outdir, "图3_双方抽奖散点.png"))
    print("已生成 3 张图")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_match.py <对局记录.csv>")
    else:
        main(sys.argv[1])
