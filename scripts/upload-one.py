"""把单个文件快速上传到 OBS（README 4B 补传场景：上传失败的大文件 / 漏传对象）。

用法（仓库根目录执行，.venv 或系统 python 均可，需已装 huaweicloudsdkobs）：
    python scripts/upload-one.py <本地文件> <OBS对象key> [--from-obs <桶内源对象key>]

例（补传基模权重到 code-dir-hhx）：
    python scripts/upload-one.py models/SmolLM2-135M-Instruct/model.safetensors code-dir-hhx/resources/model/SmolLM2-135M-Instruct/model.safetensors

例（国际线路大文件僵死时：桶内已有同内容对象，服务端复制走内网，秒级）：
    python scripts/upload-one.py models/SmolLM2-135M-Instruct/model.safetensors code-dir-hhx/resources/model/SmolLM2-135M-Instruct/model.safetensors --from-obs code-dir-v5/resources/model/SmolLM2-135M-Instruct/model.safetensors

凭证：读仓库根 .env 的 HW_AK / HW_SK / OBS_BUCKET（REGION 可选，默认 cn-north-4），
与 upload-code-dir.py 同一套。幂等：远端已有同内容（ETag==MD5）直接跳过；
失败自动重试 3 次；传完校验 ETag==本地 MD5 才算 [done]。
"""
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIES = 3
BACKOFF_S = (5, 15, 30)

from huaweicloudsdkobs.v1.obs_client import ObsClient
from huaweicloudsdkobs.v1.obs_credentials import ObsCredentials
from huaweicloudsdkobs.v1.region.obs_region import ObsRegion
from huaweicloudsdkobs.v1.model.put_object_request import PutObjectRequest
from huaweicloudsdkobs.v1.model.head_object_request import HeadObjectRequest
from huaweicloudsdkobs.v1.model.copy_object_request import CopyObjectRequest
from huaweicloudsdkcore.http.http_config import HttpConfig


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


def main():
    args = sys.argv[1:]
    copy_src = None
    if "--from-obs" in args:
        i = args.index("--from-obs")
        copy_src = args[i + 1].lstrip("/")
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    local = Path(args[0])
    key = args[1].lstrip("/")
    assert local.is_file(), f"本地文件不存在: {local}"

    env = load_env()
    for k in ("HW_AK", "HW_SK", "OBS_BUCKET"):
        if not env.get(k):
            sys.exit(f"[FATAL] .env 缺 {k}——参照 .env.example 填好")
    bucket, region = env["OBS_BUCKET"], env.get("REGION", "cn-north-4")

    http = HttpConfig()
    http.timeout = (30, 600)   # (连接, 读)超时秒——管响应等待侧僵死；发送侧僵死需外层看门狗
    obs = (ObsClient.new_builder()
           .with_http_config(http)
           .with_credentials(ObsCredentials(env["HW_AK"], env["HW_SK"]))
           .with_region(ObsRegion.value_of(region)).build())

    digest = md5_hex(local)
    size_mb = local.stat().st_size / 1e6

    # 远端已是同内容则直接跳过（幂等）
    try:
        resp = obs.head_object(HeadObjectRequest(bucket_name=bucket, object_key=key))
        etag = (getattr(resp, "e_tag", None) or "").strip('"').lower()
        if etag == digest:
            print(f"[skip] obs://{bucket}/{key} 远端已是同内容（ETag==MD5）", flush=True)
            return
    except Exception:
        pass

    # 服务端复制模式：桶内已有同内容对象（如旧版 code-dir-v*），华为云内网秒级，不过本机线路
    if copy_src:
        resp = obs.head_object(HeadObjectRequest(bucket_name=bucket, object_key=copy_src))
        src_etag = (getattr(resp, "e_tag", None) or "").strip('"').lower()
        assert src_etag == digest, \
            f"源对象 ETag={src_etag} != 本地 MD5={digest}——源与本地不是同一份文件，中止"
        print(f"[copy] obs://{bucket}/{copy_src} -> obs://{bucket}/{key}（服务端复制）", flush=True)
        obs.copy_object(CopyObjectRequest(
            bucket_name=bucket, object_key=key,
            x_obs_copy_source=f"/{bucket}/{copy_src}"))
        resp = obs.head_object(HeadObjectRequest(bucket_name=bucket, object_key=key))
        dst_etag = (getattr(resp, "e_tag", None) or "").strip('"').lower()
        assert dst_etag == digest, f"复制后 ETag 不一致: {dst_etag}"
        print(f"[done] 服务端复制完成  ETag={dst_etag}", flush=True)
        return

    print(f"[upload] {local}（{size_mb:.1f} MB）-> obs://{bucket}/{key}", flush=True)
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        try:
            with open(local, "rb") as stream:
                resp = obs.put_object(PutObjectRequest(
                    stream=stream, bucket_name=bucket, object_key=key))
            etag = (resp.e_tag or "").strip('"').lower()
            if etag != digest:
                raise RuntimeError(f"ETag 不一致 md5={digest} etag={etag}")
            print(f"[done] {time.time()-t0:.0f}s  ETag={etag}")
            return
        except KeyboardInterrupt:
            print("\n[interrupt] 直接重跑本命令续传（远端未写完整不生效，会整档重传）")
            sys.exit(130)
        except Exception as e:
            print(f"[RETRY {attempt}/{RETRIES}] {type(e).__name__}: {str(e)[:200]}"
                  f"（{BACKOFF_S[attempt-1]}s 后重试）", flush=True)
            time.sleep(BACKOFF_S[attempt - 1])
    sys.exit("[FATAL] 重试耗尽——稍后再跑一次")


if __name__ == "__main__":
    main()
