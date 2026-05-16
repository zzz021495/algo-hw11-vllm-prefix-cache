# Windows PowerShell 一鍵執行腳本: 跑兩次 vLLM 實驗 + 畫圖
$ErrorActionPreference = "Stop"

# 設定預設參數 (如果需要可自行更改模型)
$MODEL = "facebook/opt-125m"
$N = "100"
$MT = "64"

Write-Host "=== Step 1/3: enable_prefix_caching = False ===" -ForegroundColor Cyan
python experiment.py --mode off --model $MODEL --n-prompts $N --max-tokens $MT --out result_off.json

Write-Host ""
Write-Host "=== Step 2/3: enable_prefix_caching = True ===" -ForegroundColor Cyan
python experiment.py --mode on  --model $MODEL --n-prompts $N --max-tokens $MT --out result_on.json

Write-Host ""
Write-Host "=== Step 3/3: 繪圖與產出摘要 ===" -ForegroundColor Cyan
python plot_results.py

Write-Host ""
Write-Host "【成功】所有實驗已完成，結果已儲存！" -ForegroundColor Green