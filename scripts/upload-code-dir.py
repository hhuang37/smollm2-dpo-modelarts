"""把 staging/code-dir/ 整棵代码目录树上传到 OBS（GUIDE.md 阶段 5B）。

设计要点（为什么这么写）：

1. **上传对象 = 本地验证对象**。阶段 4 本地全量验证跑的就是 staging/code-dir/
   这棵树；本脚本原样整树上云——本地验证过什么，云端就跑什么，字节一致。
2. **幂等可续传**。每个对象先 head_object：远端 ETag == 本地 MD5 则跳过
   （单段 PUT 的 ETag 就是内容 MD5 hex）。大文件传到一半断了？Ctrl+C 后
   **直接重跑**——已传完的自动跳过，只续传没传完的。
3. **单对象重试**。国际线路访问国内 OBS 偶发僵死/抖动（实测同一条 269MB
   PUT 一次 13 分钟成功、一次中途僵死——线路抖动，不是硬封锁）。每个对象
   自动重试 3 次；仍失败则带清单退出，重跑即续传。
4. **终局核对**。全部传完后 list_objects 对账：逐对象比大小、查清单外多余
   对象，全对才 [done]。

凭证：读仓库根 .env 的 HW_AK / HW_SK / OBS_BUCKET（REGION 可选，默认
cn-north-4）。依赖 huaweicloudsdkobs（pip install 见 requirements-local.txt）。

用法：
    python scripts/upload-code-dir.py            # 真上传（幂等可重跑）
    python scripts/upload-code-dir.py --dry-run  # 只对照远端状态，不写云
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging" / "code-dir"
PREFIX = "code-dir/"
RETRIES = 3
BACKOFF_S = (5, 15, 30)

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


def walk_tree(base: Path):
    """staging 树 -> [(本地绝对路径, OBS key 相对 PREFIX)]，按 key 排序稳定顺序。"""
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out.append((p, p.relative_to(base).as_posix()))
    return out


def sanity_check(files):
    keys = {k for _, k in files}
    must_have = ["run_train.sh", "CODE_VERSION", "src/train_dpo.py",
                 "src/upload_outputs.py"]
    missing = [k for k in must_have if k not in keys]
    assert not missing, f"staging 树缺关键文件: {missing}——先按 GUIDE.md 阶段 3 组装"
    cv = (STAGING / "CODE_VERSION").read_text(encoding="utf-8").strip()
    assert cv, "CODE_VERSION 是空文件——阶段 3 的 git rev-parse 命令没执行成功"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出将上传/跳过的对象，不写云端")
    args = ap.parse_args()

    env = load_env()
    for k in ("HW_AK", "HW_SK", "OBS_BUCKET"):
        if not env.get(k):
            sys.exit(f"[FATAL] .env 缺 {k}——参照 .env.example 填好再跑")
    bucket = env["OBS_BUCKET"]
    region = env.get("REGION", "cn-north-4")

    assert STAGING.is_dir(), f"没有 {STAGING}——先按 GUIDE.md 阶段 3 组装代码目录"
    files = walk_tree(STAGING)
    sanity_check(files)

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

    total_mb = sum(p.stat().st_size for p, _ in files) / 1e6
    plan_upload, plan_skip = [], []
    digests = {}
    for local, key in files:
        digests[key] = md5_hex(local)
        if remote_etag(key) == digests[key]:
            plan_skip.append(key)
        else:
            plan_upload.append(key)

    for key in plan_skip:
        print(f"[skip] {PREFIX}{key}（远端已有一致版本）")
    print(f"--- 待上传 {len(plan_upload)} 个 / 跳过 {len(plan_skip)} 个 / 合计 {total_mb:.1f} MB ---")
    if args.dry_run:
        for key in plan_upload:
            print(f"[would-upload] {PREFIX}{key}")
        print("[dry-run] 未写云端。去掉 --dry-run 真上传。")
        return

    failed = []
    for local, key in files:
        if key in plan_skip:
            continue
        size = local.stat().st_size
        ok = False
        for attempt in range(1, RETRIES + 1):
            t0 = time.time()
            try:
                with open(local, "rb") as stream:
                    resp = obs.put_object(PutObjectRequest(
                        stream=stream, bucket_name=bucket, object_key=PREFIX + key))
                etag = (resp.e_tag or "").strip('"').lower()
                if etag != digests[key]:
                    raise RuntimeError(f"etag 不一致 md5={digests[key]} etag={etag}")
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

    # 终局核对：远端清单 vs 本地清单（大小逐对象 + 清单外多余对象）
    resp = obs.list_objects(ListObjectsRequest(
        bucket_name=bucket, prefix=PREFIX, max_keys=1000))
    # v1 SDK 的 contents[].size 是字符串，必须 int()（踩过）
    remote = {c.key: int(c.size) for c in (resp.contents or [])}
    mismatch = [k for _, k in files
                if remote.get(PREFIX + k) != (STAGING / k).stat().st_size]
    extra = sorted(set(remote) - {PREFIX + k for _, k in files})

    print(f"--- 终局核对：远端 {len(remote)} 对象 ---")
    for k in mismatch:
        print(f"[MISMATCH] {PREFIX}{k}: 本地 {(STAGING/k).stat().st_size} vs 远端 {remote.get(PREFIX+k)}")
    for k in extra:
        print(f"[WARN] 清单外对象: {k}")
    if failed or mismatch or extra:
        sys.exit(f"[FATAL] failed={failed} mismatch={mismatch} extra={extra}——重跑本脚本续传/核对")
    print(f"[done] {len(files)} 个对象全部在云且核对一致 -> obs://{bucket}/{PREFIX}")


if __name__ == "__main__":
    main()
