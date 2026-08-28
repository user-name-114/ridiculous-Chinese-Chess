# -*- coding: utf-8 -*-
"""对战结果可视化（原版风格 + 网络名代入 + 柱状图区间追加一档）。

用法: python plot_match.py <对局记录.csv> [名字1] [名字2]
  名字缺省：旧表头 → onnx / 纯MCTS；新表头 → P1 / P2
输出（文件名保持原版简短命名，名字体现在图内标题与图例）：
  图1_抽奖次数分布.png   图2_抽奖占比散点.png   图3_双方抽奖散点.png
  双网络对战额外输出：
  图3_抽奖次数分布2.png  图4_抽奖占比散点2.png  图5_双方抽奖散点.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

WIN_COLOR = "#2e7d32"
LOSE_COLOR = "#c62828"
DRAW_COLOR = "#9e9e9e"
NET1_COLOR = "#1a73e8"
NET2_COLOR = "#e8710a"

# 柱状图区间：在原 6 档基础上追加一档（21-30 / 31+）
LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-30", "31+"]
BOUNDS = [(0, 1), (1, 3), (3, 6), (6, 11), (11, 21), (21, 31), (31, 10 ** 9)]


def load_records(csv_path, name1, name2):
    has_p1 = None
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames
        # 新格式：列名直接使用网络名（如"修复重复虚拟损失1执子/抽奖次数"）
        has_p1 = "onnx抽奖次数" not in fns
        lot_cols = [c for c in fns if c.endswith("抽奖次数")]
        for r in reader:
            rows.append({
                "game": int(r["局号"]),
                "result": r["结果"],
                "moves": int(r["步数"]),
                "lot1": int(r[lot_cols[0]]),
                "lot2": int(r[lot_cols[1]]),
            })
    return rows, has_p1


def is_side_win(result, name):
    return result in (f"{name}胜", "P1胜", "onnx胜")


def is_side_lose(result, name):
    return result in (f"{name}负", "P1负", "onnx负")


def bin_index(n):
    return next(i for i, (lo, hi) in enumerate(BOUNDS) if lo <= n < hi)


def plot1(rows, out, name1):
    """图1：抽奖次数区间柱状图（原版样式 + 数值标注），颜色区分 name1 胜负。"""
    win = [0] * len(LABELS)
    lose = [0] * len(LABELS)
    draw = [0] * len(LABELS)
    for r in rows:
        idx = bin_index(r["lot1"])
        if is_side_win(r["result"], name1):
            win[idx] += 1
        elif is_side_lose(r["result"], name1):
            lose[idx] += 1
        else:
            draw[idx] += 1

    total = len(rows)
    win_count = sum(1 for r in rows if is_side_win(r["result"], name1))
    win_rate = win_count / total * 100 if total else 0

    x = np.arange(len(LABELS))
    w = 0.28
    plt.figure(figsize=(9, 5.5))
    bars1 = plt.bar(x - w, win, w, label=f"{name1}胜", color=WIN_COLOR)
    bars2 = plt.bar(x, lose, w, label=f"{name1}负", color=LOSE_COLOR)
    bars3 = plt.bar(x + w, draw, w, label="和", color=DRAW_COLOR)
    for bars in (bars1, bars2, bars3):
        plt.bar_label(bars, padding=2, fontsize=8)
    plt.xlabel(f"{name1} 抽奖次数区间")
    plt.ylabel("局数")
    plt.title(f"图1：{name1} 抽奖次数分布（按胜负分色）  胜率 {win_rate:.1f}%")
    plt.xticks(x, LABELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot2(rows, out, name1):
    """图2：每局抽奖占比散点（原版样式），图例带网络名。"""
    plt.figure(figsize=(8, 5))
    for r in rows:
        ratio = r["lot1"] / max(1, r["moves"]) * 100.0
        color = (WIN_COLOR if is_side_win(r["result"], name1)
                 else (LOSE_COLOR if is_side_lose(r["result"], name1)
                       else DRAW_COLOR))
        plt.scatter(r["game"], ratio, c=color, s=20, alpha=0.7)
    handles = [mpatches.Patch(color=c, label=l) for c, l in
               [(WIN_COLOR, f"{name1}胜"), (LOSE_COLOR, f"{name1}负"),
                (DRAW_COLOR, "和")]]
    plt.legend(handles=handles)
    plt.xlabel("局号")
    plt.ylabel(f"{name1} 抽奖占比（抽奖次数 / 步数，%）")
    plt.title(f"图2：{name1} 每局抽奖占比（按胜负分色）")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot3(rows, out, name1, name2, dual):
    """图3：双方抽奖数 vs 总步数散点（原版样式），图例带网络名。"""
    plt.figure(figsize=(8, 5))
    if dual:
        markers = {f"{name1}胜": "o", f"{name1}负": "x", "和": "s"}
        for r in rows:
            mk = markers.get(r["result"], "s")
            plt.scatter(r["moves"], r["lot1"], c=NET1_COLOR, marker=mk, s=22, alpha=0.6)
            plt.scatter(r["moves"], r["lot2"], c=NET2_COLOR, marker=mk, s=22, alpha=0.6)
    else:
        markers = {f"{name1}胜": "o", f"{name1}负": "x", "和": "s"}
        for r in rows:
            mk = markers.get(r["result"], "s")
            plt.scatter(r["moves"], r["lot1"], c=NET1_COLOR, marker=mk, s=22, alpha=0.6)
            plt.scatter(r["moves"], r["lot2"], c=NET2_COLOR, marker=mk, s=22, alpha=0.6)
    handles = [
        mpatches.Patch(color=NET1_COLOR, label=name1),
        mpatches.Patch(color=NET2_COLOR, label=name2),
    ]
    mhandles = [plt.Line2D([], [], color="black", marker=m, linestyle="", label=l)
                for m, l in zip(["o", "x", "s"],
                                [f"{name1}胜", f"{name1}负", "和"])]
    plt.legend(handles=handles + mhandles)
    plt.xlabel("总步数")
    plt.ylabel("抽奖次数")
    plt.title(f"图3：双方抽奖数 vs 总步数（颜色=玩家，形状=胜负，{name1}视角）")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main(csv_path, name1=None, name2=None):
    outdir = os.path.dirname(os.path.abspath(csv_path))
    rows, has_p1 = load_records(csv_path, name1, name2)
    if not rows:
        print("[错误] 对局记录为空")
        return
    if name1 is None:
        name1 = "P1" if has_p1 else "onnx"
    if name2 is None:
        name2 = ("P2" if has_p1 else "纯MCTS") if name1 in ("P1", "onnx") else "纯MCTS"

    plot1(rows, os.path.join(outdir, "图1_抽奖次数分布.png"), name1)
    plot2(rows, os.path.join(outdir, "图2_抽奖占比散点.png"), name1)
    plot3(rows, os.path.join(outdir, "图3_双方抽奖散点.png"), name1, name2,
          dual=(has_p1 and name2 not in ("纯MCTS", "P2")))
    if has_p1 and name2 not in ("纯MCTS", "P2"):
        # 双网络对战：为网络2 生成对称的两张图（胜负按 name2 视角）
        rows2 = [{"lot1": r["lot2"],
                  "result": (("P2" == "P2") and (
                      (r["result"] == "和") or
                      (r["result"] == f"{name1}胜" and False) or True))} for r in rows]
        # 重新以 name2 视角构造：name2胜 = name1负
        rows2 = []
        for r in rows:
            res2 = ("和" if r["result"] == "和"
                    else (f"{name2}胜" if r["result"] == f"{name1}负"
                          else f"{name2}负"))
            rows2.append({"lot1": r["lot2"], "result": res2, "moves": r["moves"],
                          "game": r["game"]})
        plot1(rows2, os.path.join(outdir, "图3_抽奖次数分布2.png"), name2)
        plot2(rows2, os.path.join(outdir, "图4_抽奖占比散点2.png"), name2)
        plt.figure(figsize=(8, 5))
        mk = {f"{name1}胜": "o", f"{name1}负": "x", "和": "s"}
        for r in rows:
            m = mk.get(r["result"], "s")
            plt.scatter(r["moves"], r["lot1"], c=NET1_COLOR, marker=m, s=22, alpha=0.6)
            plt.scatter(r["moves"], r["lot2"], c=NET2_COLOR, marker=m, s=22, alpha=0.6)
        handles = [mpatches.Patch(color=NET1_COLOR, label=name1),
                   mpatches.Patch(color=NET2_COLOR, label=name2)]
        mh = [plt.Line2D([], [], color="black", marker=m, linestyle="", label=l)
              for m, l in zip(["o", "x", "s"], [f"{name1}胜", f"{name1}负", "和"])]
        plt.legend(handles=handles + mh)
        plt.xlabel("总步数"); plt.ylabel("抽奖次数")
        plt.title(f"图5：双方抽奖数 vs 总步数（{name1} 视角）")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "图5_双方抽奖散点.png"), dpi=120)
        plt.close()
        print("已生成 5 张图")
    else:
        print("已生成 3 张图")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_match.py <对局记录.csv> [名字1] [名字2]")
    else:
        main(sys.argv[1],
             sys.argv[2] if len(sys.argv) > 2 else None,
             sys.argv[3] if len(sys.argv) > 3 else None)
