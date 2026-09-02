"""下载云端训练产物到本地（README.md 阶段 6.3 验收 ②）。

从 obs://<OBS_BUCKET>/<prefix> 把作业产物拉下来，保留 <run_id>/ 子目录结构
（v5 起产物按 run 分目录，重跑永不互相覆盖——下载时保留这层便于核对）。

姿势坑（SDK 实证）：新版 get_object 只接受单 request 对象，body 在
resp._stream.content——不是旧版的双参数姿势。

用法：
    python scripts/download-outputs.py --prefix outputs/dpo-run/
    # 产物落 outputs/dpo-run/<run_id>/...（保留 OBS 上的目录层级）
"""
import argparse
import re
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

from huaweicloudsdkobs.v1.obs_credentials import ObsCredentials
from huaweicloudsdkobs.v1.obs_client import ObsClient
from huaweicloudsdkobs.v1.region.obs_region import ObsRegion
from huaweicloudsdkobs.v1.model.list_objects_request import ListObjectsRequest
from huaweicloudsdkobs.v1.model.get_object_request import GetObjectRequest


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            m = re.match(r"^(\w+)=(.*)$", s)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="OBS 前缀，如 outputs/dpo-run/（= 控制台「存储训练产物」里填的路径）")
    ap.add_argument("--dest", default=None,
                    help="本地落盘根目录（默认 outputs/，保留前缀下的相对层级）")
    args = ap.parse_args()
    prefix = args.prefix if args.prefix.endswith("/") else args.prefix + "/"

    env = load_env()
    for k in ("HW_AK", "HW_SK", "OBS_BUCKET"):
        if not env.get(k):
            raise SystemExit(f"[FATAL] .env 缺 {k}——参照 .env.example 填好再跑")
    bucket = env["OBS_BUCKET"]
    region = env.get("REGION", "cn-north-4")

    dest_root = Path(args.dest) if args.dest else ROOT / "outputs"
    obs = (ObsClient.new_builder()
           .with_credentials(ObsCredentials(env["HW_AK"], env["HW_SK"]))
           .with_region(ObsRegion.value_of(region)).build())

    keys = [c.key for c in (obs.list_objects(ListObjectsRequest(
        bucket_name=bucket, prefix=prefix, max_keys=500)).contents or [])]
    files = sorted(k for k in keys if not k.endswith("/"))
    if not files:
        raise SystemExit(f"[FATAL] obs://{bucket}/{prefix} 下没有对象——作业跑完了吗？")
    print(f"obs://{bucket}/{prefix} 下 {len(files)} 个文件 -> {dest_root}")

    t_all = time.time()
    # 本地落盘 = <dest>/<前缀末段>/<run_id>/...（取前缀末段而非整段，避免 outputs\outputs\ 双层）
    local_root = dest_root / PurePosixPath(prefix.rstrip("/")).name
    for k in files:
        rel = k[len(prefix):]                       # 形如 <run_id>/eval_dpo.json
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        resp = obs.get_object(GetObjectRequest(bucket_name=bucket, object_key=k))
        dest.write_bytes(resp._stream.content)      # 新版 SDK：body 在 _stream.content
        print(f"  {dest.relative_to(dest_root)}  {dest.stat().st_size/1e6:.1f} MB  {time.time()-t0:.0f}s",
              flush=True)
    print(f"[done] {len(files)} 个文件，{time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
