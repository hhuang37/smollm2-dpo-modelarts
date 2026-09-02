#!/bin/bash
# ==============================================================================
# run_train.sh — v4 OBS「代码目录」入口脚本
#
# 定位（ADR-0005 / issue #8 实施决策 1/3）：
#   cpu-v4 镜像只装运行时环境；代码 + 基模 + 静态数据集全在 OBS 代码目录。
#   本脚本位于代码目录根部，与 src/、resources/ 同级（照抄预制模板
#   Qwen3-VL-30B-A3B-Instruct_VeRL 的 run_train_30b.sh 布局，拆解见
#   docs/research/modelarts-preset-template-teardown.md）。
#   控制台「启动命令」栏只填一条：  bash <代码目录挂载点>/run_train.sh
#
# 为什么必须是单条 sh 入口，而不是 v3 的「训练 && 自传」两段式：
#   平台超参注入 = 把超参表里的 `--名称 值` 字符串拼接到启动命令末尾，
#   且要求最后一条命令是训练脚本。两段式命令会把超参拼到自传脚本之后，
#   结构上互斥。sh 入口让超参落进 "$@"，由本脚本转发给训练脚本。
#
# 四段逻辑（每段对应一种失败模式，2026-08-31 研讨 定稿）：
#   段 1/4 必填校验     —— 缺 MODEL_PATH/DATASET 秒退，不烧 90 分钟算力才发现配错
#   段 2/4 代码目录三级兜底 —— MA_CODE_DIR → MA_JOB_DIR → 硬编码默认
#                          （官方环境变量文档只记载 MA_JOB_DIR；预制模板用的
#                           $MA_CODE_DIR 出处无记载——#11 探针 env | grep MA_
#                           实证前保持三级兜底，别提前收口）
#   段 3/4 存在性检查   —— 模型 config.json / 数据集 *.jsonl，贵操作前先失败
#   段 4/4 训练+自传    —— "$@" 转发平台超参；训练成功后调同目录自传脚本
#
# 纪律：本脚本不含任何为本地排演加的测试钩子——本地排演时自传段因平台预挂的
# MoXing whl 不存在而失败，是预期行为，如实记录即可；上传通道的真实验证是
# #11 云端探针的活（沿用 probe-v3b-upload 成熟模式）。
# ==============================================================================
set -euo pipefail

# 任何一步失败：打出失败行号与退出码（云端日志里直接定位死在哪一段）
trap 'echo "[run_train.sh] FAILED：第 $LINENO 行退出码 $?——按上方日志定位失败段" >&2' ERR

# ------------------------------------------------------------------------------
# 段 1/4：必填校验（照抄预制模板 run_train_30b.sh 的 ${VAR:?} 语法）
# 变量未设置/为空：bash 打印错误信息并立即退出（exit 1），不进入任何贵操作。
# ------------------------------------------------------------------------------
export MODEL_PATH=${MODEL_PATH:?"ERROR: MODEL_PATH 环境变量未设置——控制台「环境变量」面板应指向代码目录挂载点内的模型目录（如 .../resources/model/SmolLM2-135M-Instruct）"}
export DATASET=${DATASET:?"ERROR: DATASET 环境变量未设置——控制台「环境变量」面板应指向代码目录挂载点内的数据集目录（如 .../resources/dataset）"}

# ------------------------------------------------------------------------------
# 段 2/4：代码目录定位（三级兜底）
# 采纳标准 = 「该目录下真实存在 src/train_dpo.py」：即使某级变量被设置但指错
# 地方，也会继续向下一级兜底而不是被掩盖。每一级（含未设置）都打日志——
# #11 探针读这段日志即可钉死平台实际注入了哪个变量。
# ------------------------------------------------------------------------------
CODE_DIR=""

# 单级探测：命中（含训练脚本）返回 0；未设置或指错地方返回 1 继续下一级。
# 注意：本函数只在 || 链里被调用，set -e 不会因返回 1 而中断。
# 本脚本自身的诊断/失败输出统一走 stderr（stdout 重定向到文件时是块缓冲，
# 与无缓冲的 stderr 混入同一日志文件会时序错乱；stderr 无缓冲、保序。
# python 训练日志走 stdout，二者在 2>&1 合流后严格有序）。
try_code_dir() {
    local level="$1" cand="$2"
    if [[ -z "$cand" ]]; then
        echo "[run_train.sh] 兜底 $level：未设置/为空，跳过" >&2
        return 1
    fi
    if [[ -f "$cand/src/train_dpo.py" ]]; then
        CODE_DIR="$cand"
        echo "[run_train.sh] 兜底 $level 命中，代码目录：$CODE_DIR" >&2
        return 0
    fi
    echo "[run_train.sh] 兜底 $level：$cand 下无 src/train_dpo.py，降级" >&2
    return 1
}

try_code_dir "1/MA_CODE_DIR" "${MA_CODE_DIR:-}" \
    || try_code_dir "2/MA_JOB_DIR" "${MA_JOB_DIR:-}" \
    || try_code_dir "3/硬编码默认" "/home/ma-user/modelarts/user-job-dir" \
    || {
        echo "[run_train.sh][FATAL] 三级代码目录候选均不含 src/train_dpo.py：" >&2
        echo "  MA_CODE_DIR=${MA_CODE_DIR:-<未设置>} / MA_JOB_DIR=${MA_JOB_DIR:-<未设置>} / 硬编码=/home/ma-user/modelarts/user-job-dir" >&2
        echo "  检查控制台「代码目录」OBS 路径下是否按 issue #8 实施决策 1 布局了 src/" >&2
        exit 1
    }

# v5 可复现性（研讨 Q3 拍板）：训练容器里没有 git，git sha 靠上传时烤进
# 代码目录的 CODE_VERSION 文件（upload-code-dir.py 现场跑本机 `git rev-parse
# --short HEAD` 生成）。读出来注入 GIT_COMMIT，train_dpo.py 用它拼 run_id。
# 缺失不致命（旧目录/排演可跑），只打提示。
if [[ -f "$CODE_DIR/CODE_VERSION" ]]; then
    export GIT_COMMIT="$(tr -d '[:space:]' < "$CODE_DIR/CODE_VERSION")"
    echo "[run_train.sh][v5] CODE_VERSION -> GIT_COMMIT=$GIT_COMMIT" >&2
else
    echo "[run_train.sh][v5] $CODE_DIR/CODE_VERSION 不存在——run_id 的 git 部分将记 nogit" >&2
fi

# ------------------------------------------------------------------------------
# 段 3/4：存在性检查（贵操作前先失败）
# ------------------------------------------------------------------------------
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[run_train.sh][FATAL] MODEL_PATH 不是目录：$MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[run_train.sh][FATAL] $MODEL_PATH/config.json 缺失——MODEL_PATH 应指向完整 HF 模型目录（config.json + 权重 + tokenizer）" >&2
    exit 1
fi

# DATASET 支持两种形态：直接指向 .jsonl 文件，或指向目录（目录内必须恰好一个
# *.jsonl，MANIFEST.json 等非 jsonl 文件不受影响）。云端约定用目录形态。
if [[ -f "$DATASET" ]]; then
    DATASET_FILE="$DATASET"
elif [[ -d "$DATASET" ]]; then
    shopt -s nullglob
    jsonls=("$DATASET"/*.jsonl)
    shopt -u nullglob
    if ((${#jsonls[@]} == 0)); then
        echo "[run_train.sh][FATAL] $DATASET 下没有 *.jsonl——数据集目录应含 dpo_identity_v3.jsonl" >&2
        exit 1
    fi
    if ((${#jsonls[@]} > 1)); then
        echo "[run_train.sh][FATAL] $DATASET 下有多个 *.jsonl（${jsonls[*]}），无法决定训哪个——DATASET 直接指向文件，或目录内只留一个 jsonl" >&2
        exit 1
    fi
    DATASET_FILE="${jsonls[0]}"
else
    echo "[run_train.sh][FATAL] DATASET 既不是文件也不是目录：$DATASET" >&2
    exit 1
fi
echo "[run_train.sh] 数据集：$DATASET_FILE（$(wc -l < "$DATASET_FILE") 行）" >&2

# ------------------------------------------------------------------------------
# 段 4/4：训练（转发平台超参）→ 产物自传
# ------------------------------------------------------------------------------
TRAIN_URL=/home/ma-user/output

# 平台超参（控制台超参表：beta / lr / rpo_alpha / epochs / n_samples）以
# `--名称 值` 拼在启动命令末尾，作为 "$@" 进入本脚本。"$@" 放在命令行最后：
# argparse 同名参数后出现的生效 = 平台注入值覆盖脚本默认值。
echo "[run_train.sh] 平台超参（$# 个）：$*" >&2
python "$CODE_DIR/src/train_dpo.py" \
    --model_name "$MODEL_PATH" \
    --data_url "$DATASET_FILE" \
    --train_url "$TRAIN_URL" \
    "$@"

# 训练成功后自传（ADR-0004 MoXing 通道）：脚本读 OBS_MODEL_OUTPUT（控制台勾选
# 「存储训练产物」注入），缺失/失败都显式 Failed——显式失败优于静默丢产物。
# 本地排演时此段因平台预挂 moxing whl 不存在而失败，属预期行为。
echo "[run_train.sh] 训练完成，开始产物自传 -> OBS_MODEL_OUTPUT=${OBS_MODEL_OUTPUT:-<未设置>}" >&2
python "$CODE_DIR/src/upload_outputs.py" --train_url "$TRAIN_URL"

echo "[run_train.sh][done] 训练 + 自传全部完成" >&2
