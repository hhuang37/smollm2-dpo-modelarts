#!/usr/bin/env bash
# 阶段3实证复跑：本地全量容器训练（README 阶段 3 的 bash 等价版，删旧产物重跑）
# 预期：四段 [stage] 走完 -> 产物落 outputs/local-run/ -> 自传段非零退出（MoXing 仅云内有，预期）
set -uo pipefail
cd "$(dirname "$0")/../.."
rm -rf outputs/local-run
mkdir -p outputs/local-run

rc=0
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${PWD}:/home/ma-user/modelarts/user-job-dir/code-dir:ro" \
  -v "${PWD}/outputs/local-run:/home/ma-user/output" \
  -e MA_CODE_DIR=/home/ma-user/modelarts/user-job-dir/code-dir \
  -e MODEL_PATH=/home/ma-user/modelarts/user-job-dir/code-dir/models/SmolLM2-135M-Instruct \
  -e DATASET=/home/ma-user/modelarts/user-job-dir/code-dir/data \
  smollm2-dpo-modelarts:cpu-v1 \
  bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh \
      --beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42 || rc=$?

echo "docker run 退出码=$rc（自传段因本地无 MoXing 非零属预期；产物核验见 04 脚本）"
