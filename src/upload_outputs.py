"""训练产物自传脚本（训练完成后由 run_train.sh 调起，把产物传回 OBS）。

为什么需要它：控制台勾选「存储训练产物」只往容器注入 OBS_MODEL_OUTPUT 环境变量
（"给你张地址条"），上传是训练代码自己的责任。凭证不能用 AK/SK——容器里平台
注入的临时凭证是平台私有格式，esdk-obs-python 直签一律 403；唯一认证的消费
方式是 **MoXing**（mox.file 系列接口，容器内免配置、自动刷新凭证）。真 MoXing
不在 PyPI（同名包是占位假包），whl 由平台预挂在
/home/ma-user/modelarts/package/moxing_framework-*.whl，本脚本运行时自装。

约定：
- 目标地址读 OBS_MODEL_OUTPUT（缺失非零退出）；上传落 `<OBS_MODEL_OUTPUT>/<run_id>/`
  子目录：run_id 读 `<train_url>/RUN_ID`（train_dpo.py 训练开始时写入），重跑永不
  互相覆盖；RUN_ID 缺失退回顶层（打警告，兼容旧产物）。
- 只传 train_url 顶层文件（模型权重 + eval_dpo.json + train_log.jsonl + RUN_ID 等）；
  checkpoint-*/ 目录默认跳过（--include_checkpoints 可开）。
- 每个文件传完用 mox.file.exists 复核；任一失败 => 非零退出 => 作业状态 Failed：
  显式失败优于静默丢产物。
- 本地跑会因 MoXing whl 不存在而报错退出——预期行为（MoXing 仅训练容器内有）。
"""
import argparse
import glob
import importlib.util
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("posttrain.upload")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

MOXING_WHL_GLOB = "/home/ma-user/modelarts/package/moxing_framework-*.whl"


def ensure_moxing():
    """真 MoXing 不在 PyPI（同名是假包），从平台预挂的 whl 自装（官方 FAQ 姿势）。"""
    if importlib.util.find_spec("moxing") is not None:
        return
    wheels = glob.glob(MOXING_WHL_GLOB)
    if not wheels:
        raise RuntimeError(f"找不到 {MOXING_WHL_GLOB}——MoXing whl 由 ModelArts 平台预挂，"
                           f"本脚本只能在训练容器内运行")
    logger.info("[moxing] 未安装，自装 %s", wheels[0])
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", wheels[0]],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"moxing whl 安装失败: {r.stderr[-500:]}")
    # 同进程装包后 import 挂 No module named 'moxing' 的真实根因（2026-08-31 E2E 首跑 +
    # 本地判别实验实证）：容器内系统 site-packages 对 ma-user 无写权，pip 静默回退
    # --user 装进 ~/.local/...——该目录在解释器启动时还不存在、不在 sys.path 上，
    # invalidate_caches() 只刷新已知目录的清单缓存，救不了它，必须显式补进 sys.path；
    # 顺带 invalidate_caches() 兜住"装进已有 sys.path 目录但清单缓存过期"的情形。
    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
    importlib.invalidate_caches()
    if importlib.util.find_spec("moxing") is None:
        raise RuntimeError("whl 装完仍找不到 moxing 模块——pip 可能装到了别的 site-packages，"
                           "手动确认：python -m pip show moxing / moxing_framework")


def run(args):
    import moxing as mox   # ensure_moxing() 之后才能 import

    target = os.environ.get("OBS_MODEL_OUTPUT", "")
    if not target:
        logger.error("[FATAL] 缺 OBS_MODEL_OUTPUT 环境变量——控制台勾选「存储训练产物」"
                     "并填 obs:// 路径后，平台才会注入它；API 建作业路线不需要本脚本")
        sys.exit(1)
    target = target.rstrip("/") + "/"

    # 按 run 分子目录——RUN_ID 由 train_dpo.py 写入，重跑永不覆盖；
    # 缺失退回顶层（兼容旧产物/旧排演），但要打警告。
    run_id = ""
    run_id_path = os.path.join(args.train_url, "RUN_ID")
    if os.path.isfile(run_id_path):
        run_id = open(run_id_path, encoding="utf-8").read().strip()
    if run_id:
        target += run_id + "/"
        logger.info("[v5] 产物按 run 分子目录 -> %s", target)
    else:
        logger.warning("[v5] %s 缺失——退回顶层直传（重跑会互相覆盖；"
                       "train_dpo.py v5 起应已写入该文件）", run_id_path)

    local_files = []
    for name in sorted(os.listdir(args.train_url)):
        path = os.path.join(args.train_url, name)
        if os.path.isdir(path):
            if name.startswith("checkpoint-") and not args.include_checkpoints:
                logger.info("[skip] %s/（checkpoint 目录，--include_checkpoints 可开）", name)
                continue
            logger.info("[skip] %s/（只传顶层文件）", name)
            continue
        local_files.append((name, os.path.getsize(path)))
    if not local_files:
        logger.error("[FATAL] %s 下没有可上传的顶层文件", args.train_url)
        sys.exit(1)
    total = sum(size for _, size in local_files)
    for name, size in local_files:
        logger.info("[plan] %s（%.1f MB）", name, size / 1e6)
    logger.info("[plan] 共 %d 个文件 %.1f MB -> %s", len(local_files), total / 1e6, target)

    failed = []
    for name, size in local_files:
        dst = target + name
        t0 = time.time()
        try:
            mox.file.copy(os.path.join(args.train_url, name), dst)
            if not mox.file.exists(dst):
                raise RuntimeError("copy 返回但 exists 复核未通过")
            logger.info("[upload] %s -> %s 完成（%.1f MB，%.0fs）",
                        name, dst, size / 1e6, time.time() - t0)
        except Exception as e:
            logger.error("[upload] %s 失败：%s: %s", name, type(e).__name__, str(e)[:300])
            failed.append(name)

    if failed:
        logger.error("[FATAL] %d 个文件上传失败: %s", len(failed), failed)
        sys.exit(1)
    logger.info("[done] %d 个文件全部落 OBS（moxing 通道，零永久密钥）", len(local_files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_url", default="/home/ma-user/output",
                    help="训练产物本地目录（与 train_dpo.py 的 --train_url 一致）")
    ap.add_argument("--include_checkpoints", action="store_true",
                    help="连 checkpoint-*/ 一起上传（默认跳过省流量）")
    args = ap.parse_args()

    ensure_moxing()
    run(args)


if __name__ == "__main__":
    main()
