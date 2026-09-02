"""DPO 身份改写训练脚本（本地验证与云端训练作业的同一入口，经 run_train.sh 调起）。

流程：
  1. 训前评估：五形态 × 10 问（held-out）greedy 生成，各自称率 + 均值
  2. 读静态偏好对数据集（.jsonl，每行 {prompt, chosen, rejected}）——数据是
     离线资产（notebooks/build_dpo_dataset.ipynb 产出），作业只读文件不造数据
  3. DPO 训练（trl DPOTrainer；rpo_alpha=1.0 的 NLL 项主动抬高 chosen——纯 DPO
     会把 chosen 连带压低，实测模型退化成随机人设）
  4. 训后评估 + 保存；eval_dpo.json / train_log.jsonl / RUN_ID 写 train_url

可复现性：
  - --seed 固定全部随机源；run_id = UTC 时间戳，写 <train_url>/RUN_ID 供自传
    脚本按 run 分子目录上传（重跑永不互相覆盖）
  - dataset_fingerprint = 数据文件 MD5——验收时与本地数据比对（README 6.3）

日志：全程 logging → stdout，每优化步打 loss/rewards；异常 [FATAL] + 非零退出。
ModelArts 约定：--data_url 为输入，--train_url 为输出目录。
"""
import argparse
import hashlib
import json
import os
import sys
import time

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import (evaluate_identity_forms, load_model_and_tokenizer,
                    log_banner, logger)
# 问句底座/五形态构造/身份替换全部收口在 common.py（notebook 同源共用，防漂移）


def _load_static_pairs(path: str) -> list:
    """读静态偏好对数据集（jsonl，每行 {prompt, chosen, rejected}）。

    严格校验：三键齐全、非空字符串；坏文件当场报错，绝不静默训坏。
    """
    rows = []
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


def _file_md5(path: str) -> str:
    """数据集指纹（内容寻址，防"名字一样内容变了"的静默漂移）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_data_version(data_url: str) -> tuple:
    """(data_version, dataset_fingerprint)：version 读数据集同目录 MANIFEST.json
    （数据集的"出生证"），缺失/损坏给对应标记；fingerprint 恒为文件 MD5。
    """
    fingerprint = _file_md5(data_url)
    manifest = os.path.join(os.path.dirname(data_url) or ".", "MANIFEST.json")
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                m = json.load(f)
            return m.get("version", "static-no-version"), fingerprint
        except Exception:
            return "static-bad-manifest", fingerprint
    return "static-no-manifest", fingerprint


def run(args):
    t_start = time.time()
    os.makedirs(args.train_url, exist_ok=True)

    # --- 可复现性：seed / run_id（在任何随机行为发生之前固定） ---
    from transformers import set_seed
    set_seed(args.seed)   # random / numpy / torch / CUDA 一次全固
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    with open(os.path.join(args.train_url, "RUN_ID"), "w", encoding="utf-8") as f:
        f.write(run_id)
    logger.info("[v5] run_id=%s seed=%d", run_id, args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # --- 训前评估（分形态基线：五形态 × 10 问） ---
    logger.info("[stage] 1/4 训前基线评估（五形态 × 10 问）")
    rates_before, before_by_form, overall_before = evaluate_identity_forms(
        model, tokenizer, args.identity_name)

    # --- 数据：静态偏好对文件（离线资产，作业只读不造） ---
    logger.info("[stage] 2/4 读静态数据集 %s", args.data_url)
    rows = _load_static_pairs(args.data_url)
    if not rows:
        raise RuntimeError(f"静态数据集为空或无有效行：{args.data_url}")
    logger.info("[data] 静态数据集 %d 对（prompt/chosen/rejected 预渲染字符串）", len(rows))
    dpo_ds = Dataset.from_list(rows)

    # --- DPO 训练（rpo_alpha=1.0 抬高 chosen，防纯 DPO 连带压低；每优化步打日志） ---
    logger.info("[stage] 3/4 DPO 训练（%d 优化步）", (len(rows) + 7) // 8 * args.epochs)  # 向上取整 n/8：300 对 = 38 步（global_step 实证）
    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        rpo_alpha=args.rpo_alpha,
        num_train_epochs=args.epochs,
        seed=args.seed,                      # seed 进 config（可复现性）
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

    # --- 训练曲线落盘（trainer 现成的 log_history；生产里这条曲线接 tensorboard/MLflow） ---
    log_path = os.path.join(args.train_url, "train_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("[v5] 训练曲线 %d 条 -> %s", len(trainer.state.log_history), log_path)

    # --- 训后评估 + 保存（与训前同一口径） ---
    logger.info("[stage] 4/4 训后评估 + 保存（五形态 × 10 问）")
    rates_after, after_by_form, overall_after = evaluate_identity_forms(
        model, tokenizer, args.identity_name)
    data_version, dataset_fingerprint = _read_data_version(args.data_url)
    metrics = {
        "model": args.model_name,
        "identity_name": args.identity_name,
        "run_id": run_id,
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
    logger.info("[done] 五形态均值 %.0f%% -> %.0f%%（验收：均值 >=70%% 且 empty/none 各 >=70%%），total %.0fs",
                overall_before * 100, overall_after * 100, time.time() - t_start)
    goal_ok = (rates_after.get("empty", 0) >= 0.7
               and rates_after.get("none", 0) >= 0.7)
    if overall_after < 0.7 or not goal_ok:
        logger.warning("[done] 未过验收线：顺位——epochs 提到 2 重训；再不过 lr 提到 1e-4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_url", required=True,
                    help="静态偏好对 jsonl（run_train.sh 转发 DATASET 指向的文件）")
    ap.add_argument("--train_url", default="./outputs/dpo")
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--identity_name", default="Huang")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rpo_alpha", type=float, default=1.0)  # DPO+NLL 抬高 chosen，防纯 DPO 连带压低
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)          # 数据构造侧种子同为 42
    args = ap.parse_args()

    log_banner("train_dpo.py", vars(args))
    try:
        run(args)
    except Exception:
        logger.exception("[FATAL] train_dpo.py 失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
