# smollm2-dpo-modelarts

SmolLM2-135M-Instruct 的 DPO 偏好对齐训练，端到端跑在华为云 ModelArts（CPU）上：
从数据构造到云端训练产物验收，全程手动可复现。

**从零走一遍 → 读 [GUIDE.md](GUIDE.md)**（唯一的入口文档，7 个阶段每步有可敲命令）。

## 这条链路做什么

教一个 135M 小模型把自称改成 "Huang"（身份改写作为 DPO 的最小可验证任务）：

1. **数据**：基模自己采样生成 rejected，规则构造 chosen，产出 ~300 对偏好对；
   评估 10 问与训练问法不相交（held-out，测真泛化不测背题）；
2. **训练**：全参 DPO（trl），超参固定可复现（seed=42 + run_id + 训练曲线落盘）；
3. **上云**：纯运行时镜像推 SWR，代码+基模+数据打成 OBS 代码目录，ModelArts
   控制台手动建作业，训练完产物自动回传 OBS；
4. **验收**：产物指纹核对 + 本地 chat.py 对话 +（可选）转 GGUF 进 LM Studio
   三问——验证权重在第三方推理引擎里也站得住（¥0）。

实测：云端 CPU 作业 63 分钟 ≈ ¥0.84；训后 held-out 五形态评估 0% → 100%。

## 仓库结构

```
GUIDE.md                    ← 端到端手册（入口）
run_train.sh                ← 训练入口（容器内第一跳）
src/                        ← 训练 / 自传 / 公共库 / 对话验收
notebooks/                  ← 数据构造 notebook
docker/                     ← Dockerfile + 镜像依赖
scripts/                    ← 构建 / OBS 上传下载 / （可选）镜像中转 notebook
staging/  models/  data/  outputs/   ← gitignored 再生区（手册各阶段生成）
```

每个文件的用途见 [GUIDE.md](GUIDE.md) 开头的物料清单表。

## 前提

- 华为云账号（ModelArts + OBS + SWR）+ 一台装有 Docker 的机器；
- 云端花费 < ¥1.5/次全流程（按需计费，无长驻资源）。

## 出处

本仓库是从研究仓库 hhuang37/posttrain 的 v5 方案蒸馏出来的可复现版本
（训练代码原样，文档重组为单一线性手册）。
