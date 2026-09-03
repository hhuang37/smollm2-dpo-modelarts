#!/usr/bin/env bash
# 阶段4B实证复跑：删除远端 code-dir/ 全部对象后按 README 重传（幂等脚本全量走一遍）
set -uo pipefail
cd "$(dirname "$0")/../.."
python - <<'PYEOF'
# 删远端前缀（每个对象单删，OBS 无递归删除）
import re
env={}
for l in open(".env",encoding="utf-8"):
    s=l.strip()
    if s and not s.startswith("#"):
        m=re.match(r"^(\w+)=(.*)$",s)
        if m: env[m.group(1)]=m.group(2).strip()
from huaweicloudsdkobs.v1.obs_credentials import ObsCredentials
from huaweicloudsdkobs.v1.obs_client import ObsClient
from huaweicloudsdkobs.v1.region.obs_region import ObsRegion
from huaweicloudsdkobs.v1.model.list_objects_request import ListObjectsRequest
from huaweicloudsdkobs.v1.model.delete_object_request import DeleteObjectRequest
obs=(ObsClient.new_builder().with_credentials(ObsCredentials(env["HW_AK"],env["HW_SK"])).with_region(ObsRegion.value_of(env.get("REGION","cn-north-4"))).build())
while True:
    r=obs.list_objects(ListObjectsRequest(bucket_name=env["OBS_BUCKET"],prefix="code-dir/",max_keys=1000))
    keys=[c.key for c in (r.contents or []) if not c.key.endswith("/")]
    if not keys: break
    for k in keys:
        obs.delete_object(DeleteObjectRequest(bucket_name=env["OBS_BUCKET"],object_key=k))
    print("deleted",len(keys))
print("[remote-clean] code-dir/ 已清空")
PYEOF
python scripts/upload-code-dir.py --dry-run
python scripts/upload-code-dir.py
