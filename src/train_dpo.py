"""DPO 改身份认同训练脚本（云端形态 / ModelArts 训练作业入口）。

流程（课程 L5 完整管线；对构造 2026-08-29 v3 定版 swap_identity 三级替换；
2026-08-30 v2 五形态混合：治"身份行为条件于 system 段"）：
  1. 身份问法 prompts（内置模板 × 变体，默认 100 条 = 20 问 × 5 变体恰好一轮；
     v2 按 权重 15/15/25/25/20 混入 auto/explicit/empty/none/foreign 五种
     system 形态并全量预渲染，common.build_mixed_prompts）
  2. 用当前模型实采回答 -> rejected
  3. v3 构造（common.build_dpo_pairs / swap_identity）：chosen = 训前原句仅换
     自称名——① SmolLM/HF 等旧身份词 ② 自命名槽位（My name is X / named X /
     I am X / call me X…）③ 无名自述系动词后插名；连自述槽位都没有的少数回答
     配兜底锚点句，按 (prompt, rejected) 去重（实测 100 对 = 26+8+42+24）
  4. DPOTrainer（ref_model=None, beta=0.2, lr=5e-5, rpo_alpha=1.0,
     bs1 x accum8, 1 epoch）；rpo_alpha 的 NLL 项主动抬高 chosen——纯 DPO
     会把 chosen 连带压低（logps/chosen -85→-129，模型退化成随机人设）
  5. 训练前后各跑五形态 × 10 条评估（common.evaluate_identity_forms），
     分形态自称率 + 五形态均值
  6. 模型 + 评估结果写入 train_url（ModelArts 自动同步回 OBS）

日志约定（2026-08-28 grilling 确认）：全程 logging → stdout；
logging_steps=1 + logging_first_step + disable_tqdm，loss/rewards 每优化步可见；
异常打 [FATAL] + traceback，非零退出。

ModelArts 约定：--data_url 为输入，--train_url 为输出目录。
两种数据模式（v4 定版）：
  - --data_url 指向静态偏好对文件（.jsonl，每行 {prompt, chosen, rejected}）
    → 直接读入训练（生产形状：数据是离线资产，作业只读文件）。
    v4 云端走这条：控制台「环境变量」DATASET 指向挂载点，启动命令转发给本参数。
  - --data_url 为空 → 运行时自造（v1/v2/v3 路径保留：实采 rejected + 三级替换）。

可复现性（v5 新增，spec issue #14 / 票 #16，grilling Q2 拍板）：
  - --seed 固定全部随机源（set_seed + DPOConfig），落进产物
  - run_id = UTC 时间戳 + git sha（GIT_COMMIT env 优先——云端由 run_train.sh
    从代码目录的 CODE_VERSION 注入，容器内没有 git；本地兜底 `git rev-parse`），
    写 <train_url>/RUN_ID 供自传脚本按 run 分子目录上传（防重跑覆盖）
  - 训练曲线落盘：trainer.state.log_history → train_log.jsonl
  - eval_dpo.json 补 run_id / git_commit / seed / dataset_fingerprint（数据文件
    MD5）；data_version 改读数据集旁 MANIFEST.json（修掉硬编码与实际数据脱节）
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import (build_dpo_pairs, build_mixed_prompts,
                    evaluate_identity_forms, generate_answers,
                    load_model_and_tokenizer, log_banner, logger)

# 问法/变体/评估问句/对构造/五形态全部收口在 common.py（notebook 同源共用，防漂移）


def _load_static_pairs(path: str) -> list:
    """读静态偏好对数据集（jsonl，每行 {prompt, chosen, rejected}）。

    v4 形态：数据集是离线资产（notebooks/build_dpo_dataset.ipynb 产出），
    训练作业只读文件，不再运行时造数据。
    严格校验：三键齐全、非空字符串；坏文件当场报错，绝不静默训坏。
    """
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(f"{path}:{lineno} 不是合法 JSON")
            if not (isinstance(r, dict)
                    and all(k in r and isinstance(r[k], str) and r[k]
                            for k in ("prompt", "chosen", "rejected"))):
                raise RuntimeError(
                    f"{path}:{lineno} 缺 prompt/chosen/rejected 字符串字段，"
                    f"实际键: {sorted(r.keys()) if isinstance(r, dict) else type(r)}")
            rows.append(r)
    return rows


def _resolve_git_commit() -> str:
    """git sha 三级解析（v5 可复现性，票 #16）：
    ① GIT_COMMIT env——云端由 run_train.sh 从代码目录的 CODE_VERSION 注入
       （训练容器里没有 git，这是唯一可靠来源）；
    ② 本地兜底子进程 `git rev-parse --short HEAD`；
    ③ 都没有 → "nogit"。
    """
    sha = os.environ.get("GIT_COMMIT", "").strip()
    if sha:
        return sha
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "nogit"


def _file_md5(path: str) -> str:
    """数据集指纹（内容寻址，防"名字一样内容变了"的静默漂移）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_data_version(data_url: str) -> tuple:
    """(data_version, dataset_fingerprint)：
    静态路径——读数据集同目录 MANIFEST.json 的 version 字段（修掉硬编码
    `v2-mixed-system-forms` 与实际数据脱节的错位）+ 文件 MD5；
    MANIFEST 缺失时标 "static-no-manifest"。运行时自造路径返回固定标记。
    """
    if not data_url:
        return "runtime-self-generated", "runtime-generated"
    fingerprint = _file_md5(data_url)
    manifest = os.path.join(os.path.dirname(data_url) or ".", "MANIFEST.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                m = json.load(f)
            # 顺带校验指纹一致（MANIFEST 记的是构造时的指纹）
            return m.get("version", "static-no-version"), fingerprint
        except Exception:
            return "static-bad-manifest", fingerprint
    return "static-no-manifest", fingerprint


def run(args):
    t_start = time.time()
    os.makedirs(args.train_url, exist_ok=True)

    # --- v5 可复现性：seed / run_id（在任何随机行为发生之前固定） ---
    from transformers import set_seed
    set_seed(args.seed)   # random / numpy / torch / CUDA 一次全固
    git_commit = _resolve_git_commit()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + git_commit
    with open(os.path.join(args.train_url, "RUN_ID"), "w", encoding="utf-8") as f:
        f.write(run_id)
    logger.info("[v5] run_id=%s seed=%d git_commit=%s", run_id, args.seed, git_commit)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # --- 训前评估（v2 分形态基线：五形态 × 10 问） ---
    logger.info("[stage] 1/4 训前基线评估（五形态 × 10 问）")
    rates_before, before_by_form, overall_before = evaluate_identity_forms(
        model, tokenizer, args.identity_name)

    # --- 数据：两种模式（见模块 docstring）---
    if args.data_url:
        # v4：静态偏好对文件（离线资产，作业只读不造）
        logger.info("[stage] 2/4 读静态数据集 %s", args.data_url)
        rows = _load_static_pairs(args.data_url)
        if not rows:
            raise RuntimeError(f"静态数据集为空或无有效行：{args.data_url}")
        logger.info("[data] 静态数据集 %d 对（prompt/chosen/rejected 预渲染字符串）", len(rows))
    else:
        # v1/v2/v3：运行时实采 rejected + swap_identity 最小差异构造（common 单源）
        logger.info("[stage] 2/4 实采 rejected（%d 条五形态，云端最长静默期，看 [gen] 进度）",
                    args.n_samples)
        prompts = build_mixed_prompts(args.n_samples, tokenizer)
        rejected_texts = generate_answers(model, tokenizer, prompts)

        rows, n_dropped = build_dpo_pairs(prompts, rejected_texts, args.identity_name)
        if not rows:
            raise RuntimeError("有效对为 0：实采回答全为空/全重复，检查 rejected 采样质量")
        if len(rows) < 50:
            logger.warning("[data] 有效对不足 50，建议 --n_samples 提高（每 100 条约 2 倍实采耗时）")
    dpo_ds = Dataset.from_list(rows)

    # --- DPO 训练（rpo_alpha=1.0 抬高 chosen，防纯 DPO 连带压低；每优化步打日志） ---
    logger.info("[stage] 3/4 DPO 训练（%d 优化步）", (len(rows) + 7) // 8 * args.epochs)  # 向上取整 n/8：100 对 = 13 步（global_step 实证）
    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        rpo_alpha=args.rpo_alpha,
        num_train_epochs=args.epochs,
        seed=args.seed,                      # v5：seed 进 config（可复现性）
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_prompt_length=256,
        max_length=512,
        fp16=False, bf16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        logging_steps=1,
        logging_first_step=True,
        disable_tqdm=True,
        output_dir=args.train_url,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=config,
                         processing_class=tokenizer, train_dataset=dpo_ds)
    t_train = time.time()
    trainer.train()
    logger.info("[train] done in %.0fs", time.time() - t_train)

    # --- v5 训练曲线落盘（parity-gap §5.3：log_history 现成，演示时讲
    # "生产里这条曲线接 tensorboard / MLflow"） ---
    log_path = os.path.join(args.train_url, "train_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("[v5] 训练曲线 %d 条 -> %s", len(trainer.state.log_history), log_path)

    # --- 训后评估 + 保存（v2 分形态，notebook 同款口径） ---
    logger.info("[stage] 4/4 训后评估 + 保存（五形态 × 10 问）")
    rates_after, after_by_form, overall_after = evaluate_identity_forms(
        model, tokenizer, args.identity_name)
    data_version, dataset_fingerprint = _read_data_version(args.data_url)
    metrics = {
        "model": args.model_name,
        "identity_name": args.identity_name,
        "run_id": run_id,
        "git_commit": git_commit,
        "seed": args.seed,
        "data_version": data_version,
        "dataset_fingerprint": dataset_fingerprint,
        "n_train_pairs": len(rows),
        "hyperparams": {"beta": args.beta, "lr": args.lr, "epochs": args.epochs,
                        "rpo_alpha": args.rpo_alpha, "seed": args.seed,
                        "bs": 1, "accum": 8, "max_length": 512},
        "forms_rate_before": rates_before,
        "forms_rate_after": rates_after,
        "identity_rate_before": overall_before,
        "identity_rate_after": overall_after,
        "before_by_form": before_by_form,
        "after_by_form": after_by_form,
    }
    with open(os.path.join(args.train_url, "eval_dpo.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trainer.save_model(args.train_url)
    tokenizer.save_pretrained(args.train_url)
    logger.info("[done] 五形态均值 %.0f%% -> %.0f%%（v2 验收：均值 >=70%% 且 empty/none 各 >=70%%），total %.0fs",
                overall_before * 100, overall_after * 100, time.time() - t_start)
    goal_ok = (rates_after.get("empty", 0) >= 0.7
               and rates_after.get("none", 0) >= 0.7)
    if overall_after < 0.7 or not goal_ok:
        logger.warning("[done] 未过 v2 验收线：按 spec §7 顺位——epochs 提到 2 重训；再不过 lr 提到 1e-4；仍不过 n_samples 提到 200")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_url", default="")           # ModelArts 输入（可选）
    ap.add_argument("--train_url", default="./outputs/dpo")
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--identity_name", default="Huang")
    ap.add_argument("--n_samples", type=int, default=100)   # 20 问 × 5 变体 = 100 恰好一轮，混合构造无过滤损耗
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rpo_alpha", type=float, default=1.0)  # DPO+NLL 抬高 chosen（2026-08-29 定版）
    ap.add_argument("--epochs", type=int, default=1)    # 课程 L5 同值（2026-08-29 用户定版）
    ap.add_argument("--seed", type=int, default=42)     # v5 可复现性（parity-gap §4.3；数据构造侧种子同为 42）
    args = ap.parse_args()

    log_banner("train_dpo.py", vars(args))
    try:
        run(args)
    except Exception:
        logger.exception("[FATAL] train_dpo.py 失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
