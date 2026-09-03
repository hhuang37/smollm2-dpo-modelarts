#!/usr/bin/env bash
# 阶段4A实证复跑（自动化版）：AK/SK -> SWR 临时 docker login -> tag -> push -> digest 三方核对
# 背景：控制台复制的登录凭证 24h 过期；本脚本用
# huaweicloudsdkswr v2 CreateAuthorizationToken API 换临时凭证，全程不落明文。
set -euo pipefail
cd "$(dirname "$0")/../.."
REG=swr.cn-north-4.myhuaweicloud.com
ORG=hhuang37
SRC=smollm2-dpo-modelarts:cpu-v1
DST="$REG/$ORG/$SRC"

python - <<'PYEOF' > tests/auto-verify/.swr-login-cmd.txt
import re
env = {}
for l in open(".env", encoding="utf-8"):
    s = l.strip()
    if s and not s.startswith("#"):
        m = re.match(r"^(\w+)=(.*)$", s)
        if m: env[m.group(1)] = m.group(2).strip()
from huaweicloudsdkswr.v2.region.swr_region import SwrRegion
from huaweicloudsdkswr.v2 import SwrClient
from huaweicloudsdkswr.v2.model import CreateAuthorizationTokenRequest
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.http.http_config import HttpConfig
cfg = HttpConfig.get_default_config(); cfg.timeout = 30
client = SwrClient.new_builder().with_http_config(cfg).with_credentials(
    BasicCredentials(ak=env["HW_AK"], sk=env["HW_SK"])
).with_region(SwrRegion.value_of("cn-north-4")).build()
print(client.create_authorization_token(CreateAuthorizationTokenRequest()).x_swr_dockerlogin)
PYEOF

python - <<'PYEOF'
import re, subprocess
txt = open("tests/auto-verify/.swr-login-cmd.txt", encoding="utf-8").read().strip()
m = re.match(r"docker login -u (\S+) -p (\S+)(?: (\S+))?", txt)
user, pwd, reg = m.group(1), m.group(2), m.group(3) or "swr.cn-north-4.myhuaweicloud.com"
r = subprocess.run(["docker", "login", "-u", user, "--password-stdin", reg],
                   input=pwd, capture_output=True, text=True)
assert "Succeeded" in r.stdout + r.stderr, (r.stdout, r.stderr)
print("[login] ok")
PYEOF
rm -f tests/auto-verify/.swr-login-cmd.txt

docker tag "$SRC" "$DST"
docker push "$DST" | tee tests/auto-verify/log-05-push-swr.txt
local_digest=$(docker image inspect "$SRC" --format "{{.Descriptor.digest}}")
remote_digest=$(grep -oE "digest: sha256:[0-9a-f]+" tests/auto-verify/log-05-push-swr.txt | tail -1 | cut -d" " -f2)
echo "local=$local_digest remote=$remote_digest"
[ "$local_digest" = "$remote_digest" ] && echo "PASS: digest 三方一致" || { echo "FAIL"; exit 1; }
