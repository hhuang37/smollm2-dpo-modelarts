"""阶段5实证复跑：API 创建 ModelArts 训练作业（README 阶段 5 控制台表单的 API 等价物）。

形状与 README 阶段 5 表单 1:1（v5 直挂 code-dir 布局，同附录 A 的 dpo-run-002-api）：
  镜像      swr.cn-north-4/hhuang37/smollm2-dpo-modelarts:cpu-v1
  代码目录  obs://<OBS_BUCKET>/code-dir/
  启动命令  bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh --beta=0.2 ...
  环境变量  MODEL_PATH / DATASET / OBS_MODEL_OUTPUT（后三者勿填 MA_CODE_DIR）
  超参      beta/lr/rpo_alpha/epochs/seed（等号注入，run_train.sh "$@" 转发）

参考草稿工程 D:/work/posttrain/tools/modelarts/create-dpo-v5-console.py 改写
（去掉已废弃的 n_samples、换镜像与 code-dir 名、启动命令改字面绝对路径）。

用法（仓库根目录）:
  python tests/ai-verify/07-create-cloud-job.py            # dry-run 打印 payload
  python tests/ai-verify/07-create-cloud-job.py --create   # 真建 + 轮询到终态 + 全量日志落盘
"""
import argparse
import importlib.util as ilu
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 复用草稿工程的 client/payload 构造（BasicCredentials + ModelArtsClient + build_job）
CTJ = Path(r"D:/work/posttrain/scripts/create-training-job-api.py")
_spec = ilu.spec_from_file_location("ctj", CTJ)
ctj = ilu.module_from_spec(_spec)
_spec.loader.exec_module(ctj)

import huaweicloudsdkmodelarts.v1 as v1

NAME = "dpo-run-audit-api"
REGION = "cn-north-4"
LOCAL_CODE_DIR = "/home/ma-user/modelarts/user-job-dir"
CODE_SEG = "code-dir"           # OBS 代码目录末段 = 容器挂载目录名（README 四处同名规则）
RES = f"{LOCAL_CODE_DIR}/{CODE_SEG}/resources"
TRAIN_OBS_LOCAL = "outputs/dpo-run/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true", help="真正创建（默认 dry-run）")
    args = ap.parse_args()

    env = ctj.load_env()   # 草稿工程脚本读它自己仓库的 .env（AK/SK 与本仓库同套）
    # 本仓库 .env 优先（桶名以此为准）
    local_env = {}
    for l in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        s = l.strip()
        if s and not s.startswith("#"):
            m = re.match(r"^(\w+)=(.*)$", s)
            if m:
                local_env[m.group(1)] = m.group(2).strip()
    env.update(local_env)
    bucket = env.get("OBS_BUCKET", "posttrain")
    code_dir_obs = f"obs://{bucket}/{CODE_SEG}/"
    train_obs = f"obs://{bucket}/{TRAIN_OBS_LOCAL}"

    client = ctj.build_client(env)
    job = ctj.build_job()
    job.metadata.name = NAME
    job.metadata.description = "README 审计复跑（tests/ai-verify/07）"
    job.algorithm.engine.image_url = "hhuang37/smollm2-dpo-modelarts:cpu-v1"
    job.algorithm.code_dir = code_dir_obs
    job.algorithm.local_code_dir = LOCAL_CODE_DIR
    # 纪律：单条命令、直接以 bash <绝对路径>/run_train.sh 结尾（不包 bash -c、不 &&）
    # 超参只从这里给（README 纪律 3"别两处都填"——首跑实证：命令+parameters 双填
    # 会让平台拼参重复成 10 个，侥幸同值无害，但属违规形态）
    job.algorithm.command = (
        "bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh "
        "--beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42")
    job.algorithm.environments = {
        "MODEL_PATH": f"{RES}/model/SmolLM2-135M-Instruct",
        "DATASET": f"{RES}/dataset",
        "OBS_MODEL_OUTPUT": train_obs,   # 控制台「存储训练产物」勾选的等价物
    }
    job.algorithm.parameters = []   # 超参已在启动命令里，不重复填
    job.algorithm.outputs = None   # 控制台形态：产物纯 MoXing 自传

    print("[payload]\n" + json.dumps(job.to_dict(), ensure_ascii=False, indent=2))
    if not args.create:
        print("[dry-run] 未创建。加 --create 真建。")
        return

    resp = client.create_training_job(v1.CreateTrainingJobRequest(body=job))
    jid = resp.metadata.id
    print(f"[created] {NAME} JID={jid}", flush=True)

    phase = None
    for i in range(360):   # 20s × 360 = 120 分钟护栏
        d = client.show_training_job_details(
            v1.ShowTrainingJobDetailsRequest(training_job_id=jid))
        phase = d.status.phase
        print(f"[{i}] phase={phase}", flush=True)
        if phase in ("Completed", "Failed", "Terminated"):
            break
        time.sleep(20)
    else:
        print(f"[FATAL] 120 分钟未终态——作业保留，手动接力 JID={jid}")
        sys.exit(1)

    # 全量日志落盘 + 关键行回显
    log_path = ROOT / "tests/ai-verify/log-07-cloud-job-full.log"
    try:
        url = client.show_obs_url_of_training_job_logs(
            v1.ShowObsUrlOfTrainingJobLogsRequest(training_job_id=jid, task_id="worker-0")).obs_url
        data = urllib.request.urlopen(url, timeout=120).read()
        log_path.write_bytes(data)
        print(f"[full-log] {len(data)} B -> {log_path}")
        for line in data.decode("utf-8", "replace").splitlines():
            if any(k in line for k in ("[stage]", "五形态均值", "[done]", "[upload]",
                                       "[v5]", "run_id", "平台超参", "param",
                                       "[FATAL]", "Traceback")):
                print("  |", line[:240])
    except Exception as e:
        print(f"[full-log 拉取失败] {e}")

    print(f"[终态] {phase}（JID {jid}）")
    sys.exit(0 if phase == "Completed" else 1)


if __name__ == "__main__":
    main()
