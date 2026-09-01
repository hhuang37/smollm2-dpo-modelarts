# syntax=docker/dockerfile:1
# ==============================================================================
# 训练镜像：纯运行时（只装环境，不装任何资产）
#
# 设计原则（为什么这个镜像里"什么都没有"）：
#   ModelArts 自定义镜像作业 = 「镜像 + OBS 代码目录」两块拼图：
#     - 镜像负责"能跑"：python + torch + 训练依赖，通用且极少变化；
#     - 代码目录负责"跑什么"：训练代码 + 基模 + 数据集，随 OBS 挂载进容器。
#   好处：换模型 / 换数据 / 改代码都零重建镜像、零重推镜像（镜像推送是全链路
#   最慢的一步），只在 OBS 上传代码目录即可。
#
# 逐段说明见 GUIDE.md 阶段 2；动手改本文件前先读那段。
# ==============================================================================

# —— 基础镜像：官方 python slim。CPU 训练不需要 CUDA，slim 体积最小 ——
FROM python:3.10-slim

# —— ModelArts 约定：容器默认以 uid=1000 的非 root 用户跑（平台规范，
#    权限对不上会报各种 Permission denied）——
RUN useradd -m -u 1000 ma-user

WORKDIR /home/ma-user

# —— 依赖分两步装，各自独立成层（改业务依赖时不重装 torch）——
# 第一步：CPU 版 torch。
#   主索引必须是 pytorch 官方 CPU 源——只有它有 +cpu 构建版本；
#   再挂 PyPI 作 extra-index，供 typing-extensions / jinja2 等依赖回落
#   （单用 pytorch 索引会因依赖元数据命名不一致被新版 pip 拒收）。
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.13.0+cpu

# 第二步：训练依赖（版本全部 pin 死——"本地验证过的原样上云"，
#   浮动版本 = 某天上游发版你的作业突然挂了）。
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# —— 注意这里没有 COPY src/、没有 COPY 模型、没有 COPY 数据：
#    它们全部走 OBS 代码目录挂载，见 GUIDE.md 阶段 3 ——

USER ma-user

# HF_HUB_DISABLE_XET=1：禁用 xet 协议走经典 HTTP（hub 请求更稳；
#   本链路模型走挂载不下载，评估/分词若触发 hub 请求仍受益）。
# PYTHONUNBUFFERED=1：日志不缓冲，云端日志页实时可见。
ENV HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

# —— 不设 ENTRYPOINT / CMD：启动命令由 ModelArts 控制台传入，形如
#    bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh
#    （平台会把超参以 --name=value 拼到这条命令末尾，经 run_train.sh
#     的 "$@" 转发给训练脚本——所以镜像不能自带入口，入口在代码目录里）——
