#!/usr/bin/env bash
# Linux / WSL2 / Colab 一鍵執行:跑兩次 + 畫圖
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
N="${N:-100}"
MT="${MT:-64}"

echo "=== Step 1/3: enable_prefix_caching = False ==="
python experiment.py --mode off --model "$MODEL" --n-prompts "$N" --max-tokens "$MT" --out result_off.json

echo ""
echo "=== Step 2/3: enable_prefix_caching = True ==="
python experiment.py --mode on  --model "$MODEL" --n-prompts "$N" --max-tokens "$MT" --out result_on.json

echo ""
echo "=== Step 3/3: 繪圖比較 ==="
python plot_results.py
