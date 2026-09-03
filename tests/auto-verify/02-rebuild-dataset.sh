#!/usr/bin/env bash
# 阶段1实证复跑：删除旧数据集后无头执行 notebooks/build_dpo_dataset.ipynb --ExecutePreprocessor.kernel_name=hhxenv
# 解释器：conda hhxenv（torch 2.11+cu，镜像 pin 2.13+cpu——采样数值可能与
# 镜像侧历史数据不逐位一致，但 seed 固定、逻辑同源）
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -f data/dpo_identity_v5.jsonl data/MANIFEST.json
D:/soft/miniconda/envs/hhxenv/python.exe -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/build_dpo_dataset.ipynb --ExecutePreprocessor.kernel_name=hhxenv
echo "--- products ---"
wc -l data/dpo_identity_v5.jsonl
python -c "import json;m=json.load(open('data/MANIFEST.json',encoding='utf-8'));print('MANIFEST keys:',sorted(m))"
echo PASS
