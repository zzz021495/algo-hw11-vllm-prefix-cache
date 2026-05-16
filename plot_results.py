"""
讀取 result_off.json 與 result_on.json,畫出比較長條圖,並印出加速比。

註:vLLM V1 engine (>=0.20) 的 offline `llm.generate()` 不再回傳 per-request
metrics,所以 ttft_sec / e2e_sec 可能是 None。本腳本會自動偵測並只畫
能拿到的指標,主指標一律是「吞吐」與「總耗時」(這兩個一定有)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    off_path = Path("result_off.json")
    on_path = Path("result_on.json")
    if not off_path.exists() or not on_path.exists():
        print("找不到 result_off.json 或 result_on.json,請先跑 experiment.py 兩次。")
        sys.exit(1)

    off = load(off_path)
    on = load(on_path)

    # 一定有的核心指標
    metrics = [
        ("Throughput (req/s)", off["throughput_req_per_sec"], on["throughput_req_per_sec"], "higher better"),
        ("Throughput (tok/s)", off["throughput_tok_per_sec"], on["throughput_tok_per_sec"], "higher better"),
        ("Total elapsed (sec)", off["elapsed_sec"], on["elapsed_sec"], "lower better"),
    ]

    # 可選指標:只有 vLLM 把 per-request metrics 灌回來時才會有
    if off.get("ttft_sec") and on.get("ttft_sec"):
        metrics.append(
            ("Mean TTFT (sec)", off["ttft_sec"]["mean"], on["ttft_sec"]["mean"], "lower better")
        )
    if off.get("e2e_sec") and on.get("e2e_sec"):
        metrics.append(
            ("Mean E2E latency (sec)", off["e2e_sec"]["mean"], on["e2e_sec"]["mean"], "lower better")
        )

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (title, v_off, v_on, hint) in zip(axes, metrics):
        bars = ax.bar(["caching=off", "caching=on"], [v_off, v_on],
                      color=["#888888", "#2a9d8f"])
        ax.set_title(f"{title}\n({hint})")
        ax.bar_label(bars, fmt="%.3g", padding=3)
        ax.margins(y=0.2)

    fig.suptitle(
        f"vLLM Prefix Caching:off vs. on   (model={off['model']}, n={off['n_prompts']})",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = "comparison.png"
    fig.savefig(out_png, dpi=130)
    print(f"[saved] {out_png}")

    # 文字摘要
    speedup_req = on["throughput_req_per_sec"] / off["throughput_req_per_sec"]
    speedup_tok = on["throughput_tok_per_sec"] / off["throughput_tok_per_sec"]
    elapsed_red = 1 - on["elapsed_sec"] / off["elapsed_sec"]

    print("\n========== 比較摘要 ==========")
    print(f"  吞吐 (req/s) 加速:  {speedup_req:.2f} 倍")
    print(f"  吞吐 (tok/s) 加速:  {speedup_tok:.2f} 倍")
    print(f"  總耗時下降:         {elapsed_red * 100:5.1f}%  "
          f"({off['elapsed_sec']:.2f}s → {on['elapsed_sec']:.2f}s)")

    if off.get("ttft_sec") and on.get("ttft_sec"):
        ttft_red = 1 - on["ttft_sec"]["mean"] / off["ttft_sec"]["mean"]
        print(f"  平均 TTFT 下降:     {ttft_red * 100:5.1f}%")
    else:
        print("  (TTFT 缺失:vLLM V1 engine 在 offline 模式不回傳 per-request metrics)")

    if off.get("e2e_sec") and on.get("e2e_sec"):
        e2e_red = 1 - on["e2e_sec"]["mean"] / off["e2e_sec"]["mean"]
        print(f"  平均 E2E latency 下降: {e2e_red * 100:5.1f}%")

    print("\n對照投影片宣稱「吞吐 +2~4x、延遲 -30~50%」,確認觀測到類似量級即達成實驗目標。")


if __name__ == "__main__":
    main()
