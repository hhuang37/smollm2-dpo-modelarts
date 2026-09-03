#!/usr/bin/env bash
# 阶段0.4实证复跑（修正版）：显式下载基模 5 个文件并校验
# 坑：
#   1) huggingface-cli 已死 -> hf download
#   2) 整仓下载拉 25 个文件 -> 显式列 5 个
#   3) hf-mirror + hub 1.18 + stored token 报 FileMetadataError -> 直连 + 禁 xet
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
M=models/SmolLM2-135M-Instruct
rm -rf "$M"
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json \
  --local-dir "$M"
echo "--- files ---"
ls "$M"
n=$(ls "$M" | wc -l)
[ "$n" -eq 5 ] || { echo "FAIL: expect 5 files, got $n"; exit 1; }
echo "PASS: 5 files"
