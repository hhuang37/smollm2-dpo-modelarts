#!/usr/bin/env bash
# 验收④实证复跑（引擎级裸测自动化版）：训后权重转 GGUF，再用 llama-cli 直喂 chatml
# 问名字（等价 LM Studio 三问的引擎侧证据，README 6.5 排障定位器同款）。
# 用法: bash tests/ai-verify/09-gguf-engine-test.sh <权重目录>   # 如 outputs/local-run
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${1:?用法: 09-gguf-engine-test.sh <训后权重目录>}
mkdir -p outputs/lmstudio
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${PWD}/${SRC}:/models/in:ro" -v "${PWD}/outputs/lmstudio:/models/out" \
  ghcr.io/ggml-org/llama.cpp:full \
  --convert /models/in --outfile /models/out/audit-f16.gguf
printf '<|im_start|>user\nwhat is your name<|im_end|>\n<|im_start|>assistant\n' > outputs/lmstudio/p1.txt
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /app/llama-cli \
  -v "${PWD}/outputs/lmstudio:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:full \
  -m /models/audit-f16.gguf -f /models/p1.txt -n 40 --temp 0 -c 512 \
  --no-display-prompt --single-turn | tee outputs/lmstudio/llama-cli-out.txt
grep -qi huang outputs/lmstudio/llama-cli-out.txt && echo "PASS: llama.cpp 引擎答 Huang" || { echo "FAIL: 未答 Huang"; exit 1; }
