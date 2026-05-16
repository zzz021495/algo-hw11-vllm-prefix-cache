"""
vLLM Prefix Caching 體驗實驗 — Chapter 11 / p.76
=====================================================
這支腳本會用 100 個「共用同一段長前綴」的 prompt 一次餵給 vLLM,
分別在 enable_prefix_caching=True / False 兩種設定下量測:

  • 端到端吞吐  (req/s, tokens/s)
  • TTFT        (Time-to-First-Token,平均、p50、p95)
  • E2E latency (每個 request 從 arrival 到 finished 的耗時)

設計重點:
  1. 共用前綴刻意做到 ~800 tokens,確保大過 vLLM 預設 16-token block,
     prefix hash 命中後可以省掉大量 forward 計算。
  2. 100 個 prompt 後綴各自不同(不同算式),確保只有「前綴」可重用,
     不會作弊讓整段都命中。
  3. SamplingParams 設 temperature=0、ignore_eos=True、固定 max_tokens,
     讓兩次跑出同樣的 token 數,比較才公平。

用法 (在 Linux / WSL / Colab 環境):
    python experiment.py --mode off --out result_off.json
    python experiment.py --mode on  --out result_on.json
    python plot_results.py
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

# ---------------------------------------------------------------------------
# 共用長前綴:system prompt + 5 個 few-shot 範例,刻意湊到 ~800 tokens。
# 100 個 prompt 都會共用這段,所以 prefix caching 開啟時會大量命中。
# ---------------------------------------------------------------------------
SHARED_PREFIX = """You are a meticulous mathematics tutor working with first-year
university students. You explain every arithmetic step in detail, never skip
intermediate computations, and always double-check the final answer before
presenting it. You follow these rules without exception:

  Rule 1. Always restate the problem in your own words first.
  Rule 2. Identify which arithmetic operations are needed and in what order,
          following the standard precedence (parentheses, exponents,
          multiplication and division left-to-right, then addition and
          subtraction left-to-right).
  Rule 3. Show every intermediate value on its own line so that the student
          can follow along.
  Rule 4. After producing the answer, verify it by performing a quick sanity
          check (rough magnitude estimate or inverse operation).
  Rule 5. End your response with a single line of the form `FINAL: <number>`.

Below are five worked examples that demonstrate the expected style.

Example 1.
Q: What is 12 * 7 + 5?
A: We need 12 * 7 first, then add 5.
   12 * 7 = 84
   84 + 5 = 89
   Sanity check: 12 * 7 is close to 10 * 7 = 70, plus a bit more, so 84 is
   reasonable; adding 5 gives 89.
   FINAL: 89

Example 2.
Q: What is 23 * 11 + 4?
A: First compute 23 * 11.
   23 * 11 = 23 * 10 + 23 = 230 + 23 = 253
   253 + 4 = 257
   Sanity check: 23 * 11 should be slightly more than 23 * 10 = 230, and 253
   matches; adding 4 gives 257.
   FINAL: 257

Example 3.
Q: What is 9 * 13 + 2?
A: 9 * 13 = 9 * 10 + 9 * 3 = 90 + 27 = 117
   117 + 2 = 119
   Sanity check: 9 * 13 is close to 10 * 13 = 130, minus 13, so 117; plus 2
   gives 119.
   FINAL: 119

Example 4.
Q: What is 18 * 6 + 11?
A: 18 * 6 = (20 - 2) * 6 = 120 - 12 = 108
   108 + 11 = 119
   Sanity check: 18 * 6 should be close to 20 * 6 = 120, slightly less, and
   108 fits; plus 11 gives 119.
   FINAL: 119

Example 5.
Q: What is 25 * 8 + 7?
A: 25 * 8 = 200
   200 + 7 = 207
   Sanity check: 25 * 8 is exactly a quarter of 800, which is 200; plus 7
   gives 207.
   FINAL: 207

Now answer the next question using exactly the same step-by-step style.
"""


def make_prompts(n: int) -> list[str]:
    """產生 n 個共用前綴、後綴各異的 prompt。"""
    prompts = []
    for i in range(n):
        a = 17 + i
        b = 31 + (i % 7)
        c = i
        question = f"What is {a} * {b} + {c}?"
        prompts.append(SHARED_PREFIX + f"\nQ: {question}\nA:")
    return prompts


def _stat(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return {
        "mean": mean(xs_sorted),
        "p50": xs_sorted[n // 2],
        "p95": xs_sorted[min(n - 1, int(n * 0.95))],
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
    }


def run(
    enable_prefix_caching: bool,
    model: str,
    n_prompts: int,
    max_tokens: int,
    gpu_mem_util: float,
    max_model_len: int,
) -> dict:
    # 延後 import,讓 --help 在沒裝 vllm 的機器也能跑。
    from vllm import LLM, SamplingParams

    print(
        f"[init] model={model}  enable_prefix_caching={enable_prefix_caching}  "
        f"n_prompts={n_prompts}  max_tokens={max_tokens}"
    )

    llm = LLM(
        model=model,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="auto",
        enforce_eager=False,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,  # 強制吐滿 max_tokens,讓兩次跑的工作量一致
    )

    prompts = make_prompts(n_prompts)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    in_tok = sum(len(o.prompt_token_ids) for o in outputs)
    out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)

    ttfts: list[float] = []
    e2es: list[float] = []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if m is None:
            continue
        if m.first_token_time and m.arrival_time:
            ttfts.append(m.first_token_time - m.arrival_time)
        if m.finished_time and m.arrival_time:
            e2es.append(m.finished_time - m.arrival_time)

    result = {
        "enable_prefix_caching": enable_prefix_caching,
        "model": model,
        "n_prompts": n_prompts,
        "max_tokens": max_tokens,
        "elapsed_sec": elapsed,
        "throughput_req_per_sec": n_prompts / elapsed,
        "throughput_tok_per_sec": (in_tok + out_tok) / elapsed,
        "input_tokens_total": in_tok,
        "output_tokens_total": out_tok,
        "input_tokens_per_prompt": in_tok // n_prompts,
        "ttft_sec": _stat(ttfts),
        "e2e_sec": _stat(e2es),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["on", "off"], required=True,
                   help="on = enable_prefix_caching=True;off = False")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace 模型名稱。GPU 太小可改 facebook/opt-125m")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gpu-mem-util", type=float, default=0.5,
                   help="vLLM 預先佔用的 GPU 記憶體比例")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--out", default="result.json")
    args = p.parse_args()

    enable = (args.mode == "on")
    res = run(
        enable_prefix_caching=enable,
        model=args.model,
        n_prompts=args.n_prompts,
        max_tokens=args.max_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n========== 結果 ==========")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
