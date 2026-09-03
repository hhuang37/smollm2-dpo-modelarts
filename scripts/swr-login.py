"""用 AK/SK 换 SWR 临时 docker login 凭证并执行登录（README 阶段 4A）。

为什么需要它：控制台「客户端上传」复制的登录指令是 24h 临时令牌，过期后 push
报 `Authenticate Error - Get user token error`——本脚本用 .env 的 AK/SK 调 SWR
CreateAuthorizationToken API 现场换新令牌并 docker login，token 全程不落盘、
不回显（CI / agent 友好；2026-09-02 实测）。

凭证：读仓库根 .env 的 HW_AK / HW_SK（REGION 可选，默认 cn-north-4）。
依赖：pip install huaweicloudsdkswr（requirements-local.txt 已含）。

用法（仓库根目录）：
    python scripts/swr-login.py          # 换令牌并 docker login
    python scripts/swr-login.py --print  # 只打印登录指令（人肉复制用）
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.http.http_config import HttpConfig
from huaweicloudsdkswr.v2 import SwrClient
from huaweicloudsdkswr.v2.model import CreateAuthorizationTokenRequest
from huaweicloudsdkswr.v2.region.swr_region import SwrRegion

DEFAULT_REGION = "cn-north-4"


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
    ap.add_argument("--print", action="store_true", dest="print_cmd",
                    help="只打印 docker login 指令（含临时密码），不执行")
    args = ap.parse_args()

    env = load_env()
    for k in ("HW_AK", "HW_SK"):
        if not env.get(k):
            sys.exit(f"[FATAL] .env 缺 {k}——参照 .env.example 填好再跑")
    region = env.get("REGION", DEFAULT_REGION)
    registry = f"swr.{region}.myhuaweicloud.com"

    cfg = HttpConfig.get_default_config()
    cfg.timeout = 30
    client = (SwrClient.new_builder()
              .with_http_config(cfg)
              .with_credentials(BasicCredentials(ak=env["HW_AK"], sk=env["HW_SK"]))
              .with_region(SwrRegion.value_of(region))
              .build())
    # 返回的是完整 docker login 指令（用户名/密码即临时凭证）
    cmd = client.create_authorization_token(
        CreateAuthorizationTokenRequest()).x_swr_dockerlogin
    m = re.match(r"docker login -u (\S+) -p (\S+)(?: (\S+))?", cmd)
    if not m:
        sys.exit(f"[FATAL] 登录指令格式异常：{cmd[:30]}...")
    user, pwd, reg = m.group(1), m.group(2), m.group(3) or registry

    if args.print_cmd:
        print(cmd)
        return
    r = subprocess.run(["docker", "login", "-u", user, "--password-stdin", reg],
                       input=pwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if "Succeeded" not in out:
        sys.exit(f"[FATAL] docker login 失败：{out.strip()}")
    print(f"[done] 已登录 {reg}（临时凭证，24h 内有效）")


if __name__ == "__main__":
    main()
