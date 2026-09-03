#!/usr/bin/env bash
# 阶段6实证复跑：下载云端产物 -> 14 对象核对 -> 指纹核对 -> chat.py 对话验收
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/download-outputs.py --prefix outputs/dpo-run/
run_id=$(ls outputs/dpo-run | sort | tail -1)
echo "latest run_id: $run_id"
n=$(ls "outputs/dpo-run/$run_id" | wc -l)
echo "objects: $n"; [ "$n" -eq 14 ] || echo "WARN: 期望 14 个顶层文件，实际 $n"
python - "$run_id" <<'PYEOF'
import json, hashlib, sys
run_id = sys.argv[1]
e = json.load(open(f"outputs/dpo-run/{run_id}/eval_dpo.json", encoding="utf-8"))
local = hashlib.md5(open("data/dpo_identity_v5.jsonl", "rb").read()).hexdigest()
print("cloud fp:", e["dataset_fingerprint"], "| local md5:", local,
      "| before:", e["identity_rate_before"], "| after:", e["identity_rate_after"])
assert e["dataset_fingerprint"] == local, "指纹不一致"
assert e["identity_rate_after"] >= 0.7
print("PASS: 指纹一致且训后达标")
PYEOF
D:/soft/miniconda/envs/hhxenv/python.exe src/chat.py --model "outputs/dpo-run/$run_id" --prompt "Who are you?"
