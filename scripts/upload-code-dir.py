"""把训练物料从仓库源位置直接上传到 OBS code-dir/（README.md 阶段 4B）。

没有"组装/staging"中间层：云端代码目录的固定布局就是下面这张 源文件→OBS key
映射表（恰好 11 个对象），本脚本逐对象直传。

    本机源                                      OBS key（PREFIX = code-dir/）
    -------------------------------------------------------------------
    run_train.sh                               run_train.sh
    src/{train_dpo,common,upload_outputs}.py   src/...
    models/SmolLM2-135M-Instruct/（5 文件）     resources/model/SmolLM2-135M-Instruct/...
    data/{dpo_identity_v5.jsonl,MANIFEST.json} resources/dataset/...

设计要点：

1. **幂等可续传**。每个对象先 head_object：远端 ETag == 本地 MD5 则跳过
   （单段 PUT 的 ETag 就是内容 MD5 hex）。大文件传到一半断了？Ctrl+C 后
   **直接重跑**——已传完的自动跳过，只续传没传完的。
2. **单对象重试**。国际线路访问国内 OBS 偶发僵死/抖动。每个对象自动重试
   3 次；仍失败则带清单退出，重跑即续传。
3. **终局核对**。全部传完后 list_objects 对账：逐对象比大小、查清单外多余
   对象，全对才 [done]。
4. **改了代码/数据 → 重跑即可**。脚本永远读当前仓库内容——重跑等于把最新
   状态推上云。

凭证：读仓库根 .env 的 HW_AK / HW_SK / OBS_BUCKET（REGION 可选，默认
cn-north-4）。依赖 huaweicloudsdkobs（pip install 见 requirements-local.txt）。

用法：
    python scripts/upload-code-dir.py                          # 真上传（幂等可重跑）
    python scripts/upload-code-dir.py --dry-run                # 只对照远端状态，不写云
    python scripts/upload-code-dir.py --prefix code-dir-hhx/   # 目标文件夹带轮次后缀时
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "code-dir/"
RETRIES = 3
BACKOFF_S = (5, 15, 30)

MODEL_DIR = ROOT / "models" / "SmolLM2-135M-Instruct"
# HF 源原样的 5 件，一个不能少（缺 config/tokenizer 容器里加载即失败）
MODEL_FILES = ["model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "config.json", "generation_config.json"]
SRC_FILES = ["train_dpo.py", "common.py", "upload_outputs.py"]
DATASET_FILES = ["dpo_identity_v5.jsonl", "MANIFEST.json"]

from huaweicloudsdkobs.v1.obs_credentials import ObsCredentials
from huaweicloudsdkobs.v1.obs_client import ObsClient
from huaweicloudsdkobs.v1.region.obs_region import ObsRegion
from huaweicloudsdkobs.v1.model.put_object_request import PutObjectRequest
from huaweicloudsdkobs.v1.model.head_object_request import HeadObjectRequest
from huaweicloudsdkobs.v1.model.list_objects_request import ListObjectsRequest


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            m = re.match(r"^(\w+)=(.*)$", s)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_plan():
    """[(本地路径, OBS key, MD5)]——布局映射的代码形态。源缺了报人话退出。"""
    if not (ROOT / "run_train.sh").is_file():
        sys.exit("[FATAL] 仓库根没有 run_train.sh——本脚本要在本仓库内运行")
    for f in SRC_FILES:
        if not (ROOT / "src" / f).is_file():
            sys.exit(f"[FATAL] 缺 src/{f}——仓库不完整，重新 clone")
    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).is_file()]
    if missing:
        sys.exit(f"[FATAL] 基模缺 {missing}——先跑阶段 0.4：huggingface-cli download "
                 "HuggingFaceTB/SmolLM2-135M-Instruct")
    for f in DATASET_FILES:
        if not (ROOT / "data" / f).is_file():
            sys.exit(f"[FATAL] 缺 data/{f}——先跑阶段 1 的 notebooks/build_dpo_dataset.ipynb")

    pairs = [(ROOT / "run_train.sh", "run_train.sh")]
    pairs += [(ROOT / "src" / f, f"src/{f}") for f in SRC_FILES]
    pairs += [(MODEL_DIR / f, f"resources/model/SmolLM2-135M-Instruct/{f}")
              for f in MODEL_FILES]
    pairs += [(ROOT / "data" / f, f"resources/dataset/{f}") for f in DATASET_FILES]

    return [(p, k, md5_hex(p)) for p, k in pairs]


def main():
    global PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出将上传/跳过的对象，不写云端")
    ap.add_argument("--prefix", default=PREFIX,
                    help=f"OBS 目标前缀（默认 {PREFIX}；目标文件夹带轮次后缀时传如 code-dir-hhx/）")
    args = ap.parse_args()
    PREFIX = args.prefix.rstrip("/") + "/"   # 归一化：必以 / 结尾

    env = load_env()
    for k in ("HW_AK", "HW_SK", "OBS_BUCKET"):
        if not env.get(k):
            sys.exit(f"[FATAL] .env 缺 {k}——参照 .env.example 填好再跑")
    bucket = env["OBS_BUCKET"]
    region = env.get("REGION", "cn-north-4")

    files = build_plan()

    obs = (ObsClient.new_builder()
           .with_credentials(ObsCredentials(env["HW_AK"], env["HW_SK"]))
           .with_region(ObsRegion.value_of(region)).build())

    def remote_etag(key: str):
        """远端对象 ETag；不存在返回 None。head 404 不报错（走 status）。"""
        try:
            resp = obs.head_object(HeadObjectRequest(
                bucket_name=bucket, object_key=PREFIX + key))
            etag = (getattr(resp, "e_tag", None) or "").strip('"').lower()
            return etag or None
        except Exception:
            return None

    print(f"=== 代码目录上传 -> obs://{bucket}/{PREFIX}（{len(files)} 个对象，"
          f"region {region}{', DRY-RUN' if args.dry_run else ''}） ===")

    total_mb = sum(p.stat().st_size for p, _, _ in files) / 1e6
    plan_upload, plan_skip = [], []
    for local, key, digest in files:
        (plan_skip if remote_etag(key) == digest else plan_upload).append(key)

    for key in plan_skip:
        print(f"[skip] {PREFIX}{key}（远端已有一致版本）")
    print(f"--- 待上传 {len(plan_upload)} 个 / 跳过 {len(plan_skip)} 个 / 合计 {total_mb:.1f} MB ---")
    if args.dry_run:
        for key in plan_upload:
            print(f"[would-upload] {PREFIX}{key}")
        print("[dry-run] 未写云端。去掉 --dry-run 真上传。")
        return

    failed = []
    for local, key, digest in files:
        if key in plan_skip:
            continue
        size = local.stat().st_size
        ok = False
        for attempt in range(1, RETRIES + 1):
            t0 = time.time()
            try:
                with open(local, "rb") as stream:
                    resp = obs.put_object(PutObjectRequest(
                        stream=stream, bucket_name=bucket,
                        object_key=PREFIX + key))
                etag = (resp.e_tag or "").strip('"').lower()
                if etag != digest:
                    raise RuntimeError(f"etag 不一致 md5={digest} etag={etag}")
                print(f"[OK] {PREFIX}{key}  {size/1e6:.1f} MB  {time.time()-t0:.0f}s")
                ok = True
                break
            except KeyboardInterrupt:
                print(f"\n[interrupt] {key} 中断——已传完的对象不受影响，直接重跑本脚本续传")
                sys.exit(130)
            except Exception as e:
                print(f"[RETRY {attempt}/{RETRIES}] {key}: {type(e).__name__}: "
                      f"{str(e)[:200]}（{BACKOFF_S[attempt-1]}s 后重试）", flush=True)
                time.sleep(BACKOFF_S[attempt - 1])
        if not ok:
            failed.append(key)

    # 终局核对：远端清单 vs 计划清单（大小逐对象 + 清单外多余对象）
    resp = obs.list_objects(ListObjectsRequest(
        bucket_name=bucket, prefix=PREFIX, max_keys=1000))
    # v1 SDK 的 contents[].size 是字符串，必须 int()（踩过）
    remote = {c.key: int(c.size) for c in (resp.contents or [])}
    local_sizes = {k: p.stat().st_size for p, k, _ in files}
    mismatch = [k for k in local_sizes if remote.get(PREFIX + k) != local_sizes[k]]
    # 远端键带 PREFIX、计划键是裸键，比对前先对齐（否则全部误报清单外）
    extra = sorted(set(remote) - {PREFIX + k for k in local_sizes})

    print(f"--- 终局核对：远端 {len(remote)} 对象 ---")
    for k in mismatch:
        print(f"[MISMATCH] {PREFIX}{k}: 本地 {local_sizes[k]} vs 远端 {remote.get(PREFIX+k)}")
    for k in extra:
        print(f"[WARN] 清单外对象: {k}")
    if failed or mismatch or extra:
        sys.exit(f"[FATAL] failed={failed} mismatch={mismatch} extra={extra}——重跑本脚本续传/核对")
    print(f"[done] {len(files)} 个对象全部在云且核对一致 -> obs://{bucket}/{PREFIX}")


if __name__ == "__main__":
    main()
