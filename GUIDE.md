# 端到端实操手册：SmolLM2-135M DPO 训练上华为 ModelArts

> 按本手册从零走一遍：**构造数据 → 构建镜像 → 组装代码目录 → （可选）本地全量验证 →
> 上传 → 控制台建作业 → 云端训练 + 产物回传验收**。全程手动、每步有可敲命令。
> 实测参照：云端 CPU 作业 63 分钟 ≈ ¥0.84；全流程云端花费 < ¥1.5。
>
> 命令以 Windows PowerShell 为主（Git Bash 用户注意 `MSYS_NO_PATHCONV=1`，
> 防止 `/home/...` 容器路径被改写成 Windows 路径）。

---

## 全链路地图

读法：**自上而下 = 阶段 1→7 的执行顺序**；【】标注该步在哪头执行；`├──>` 指向产物在云端的落点；阶段 4 是可跳过的本地验证支线。

```text
阶段1【本机】notebooks/build_dpo_dataset.ipynb
             基模采样 rejected + 规则构造 chosen → data/dpo_identity_v5.jsonl
                        │
阶段2【本机】docker build 纯运行时镜像 smollm2-dpo-modelarts:cpu-v1（只装依赖）
                        │
阶段3【本机】组装 staging/code-dir/（代码 + 基模 + 数据 + CODE_VERSION）
                        │
                        ├─（可选）阶段4：docker run 同镜像 + 同代码目录本地全量训练
                        │            产物落 outputs/local-run/，chat.py 顺手验收
                        │
阶段5A【本机→云】docker push 推镜像（直推，或 Docker Hub + crane 中转）
                        ├──────> SWR 镜像仓库
阶段5B【本机→云】upload-code-dir.py 代码目录整树上传（幂等续传）
                        ├──────> OBS obs://<桶>/code-dir/
                        │
阶段6【云端】ModelArts 控制台创建训练作业（镜像 = SWR，代码目录 = OBS code-dir）
                        │
      【云端】容器内 bash run_train.sh：校验 → 定位 → 检查 → DPO 训练 → 训前/训后 held-out 评估
                        │
      【云端】MoXing 产物自传（按 run_id 分子目录）
                        ├──────> OBS obs://<桶>/outputs/dpo-run/<run_id>/
                        │
阶段7【本机】download-outputs.py 下载产物：指纹核对 + chat.py 对话验收
```

三层分层（本方案的核心架构主张）：

| 层 | 放什么 | 在哪 |
|---|---|---|
| 镜像 | 只装运行环境（python + torch + 依赖） | SWR，极少变化 |
| 代码目录 | 训练代码 + 基模 + 数据集 + CODE_VERSION | OBS `code-dir/`，每次变更重传 |
| 作业配置 | 路径环境变量 + 超参 + 产物地址 | ModelArts 控制台表单 |

换模型 / 换数据 / 改代码：**零重建镜像**，只重传代码目录；换依赖才重建镜像。

---

## 物料清单（仓库里每个文件是干什么的）

| 文件 | 作用 | 阶段 |
|---|---|---|
| `run_train.sh` | 训练入口（容器内第一跳）：必填校验→代码目录定位→资产存在性检查→训练+产物自传。控制台「启动命令」指向它 | 3/4/6 |
| `src/train_dpo.py` | DPO 训练主脚本：读静态数据集训练，训前/训后各跑一次 held-out 评估，写 `eval_dpo.json`/`train_log.jsonl`/`RUN_ID` | 3/4/6 |
| `src/common.py` | 公共库：模型加载、20 训练问底座、评估 10 问、五形态 prompt 构造 | 1/3/4 |
| `src/upload_outputs.py` | 产物自传（仅容器内可用）：走平台预挂的 MoXing 通道上传到 OBS，按 run_id 分子目录 | 3/6 |
| `src/chat.py` | 本地对话验收工具：加载训练前后模型问一句看效果 | 4/7 |
| `notebooks/build_dpo_dataset.ipynb` | 数据构造：基模采样生成 rejected + 规则构造 chosen，产出训练 jsonl + MANIFEST | 1 |
| `docker/Dockerfile` | 纯运行时镜像定义（逐段教学注释见文件内） | 2 |
| `docker/requirements.txt` | 镜像内训练依赖（版本全部 pin 死） | 2 |
| `requirements-local.txt` | 本机环境依赖（数据构造 + 对话验收 + OBS 脚本），版本与镜像内一致 | 0 |
| `scripts/build-image.ps1` | 构建 + 冒烟一键化（`--provenance=false` + manifest 断言 + import 冒烟） | 2 |
| `scripts/upload-code-dir.py` | 代码目录整树上 OBS：幂等续传 + 重试 + 终局核对，`--dry-run` 可预演 | 5B |
| `scripts/download-outputs.py` | 云端产物下载（保留 run_id 目录层级） | 7 |
| `scripts/relay-image-to-swr.ipynb` | 镜像中转（**可选**）：本机在国外时，经 Docker Hub 中转把镜像搬进 SWR | 5A |
| `.env.example` | 凭证与桶名模板（复制为 `.env`，gitignored） | 0 |
| `.gitattributes` | 钉死 `*.sh` 行尾为 LF（Windows autocrlf 会让容器里 bash 报错） | — |
| `.dockerignore` | 构建上下文只留 requirements（构建快、防大文件进镜像） | 2 |

---

## 阶段 0：前置准备

### 0.1 账号与云上资源

| # | 需要什么 | 哪里开 | 说明 |
|---|---|---|---|
| 1 | 华为云账号（实名认证） | huaweicloud.com | 国际站/中国站均可，本手册以中国站 cn-north-4 为例 |
| 2 | ModelArts 服务 | 控制台开通 | 按需计费，不用不花钱 |
| 3 | OBS 桶 | 控制台 OBS 创建桶 | 例：`posttrain`（区域 cn-north-4，标准存储） |
| 4 | SWR 组织 | 控制台 SWR → 组织管理 | 例：你的用户名。镜像要推到这里 |
| 5 | AK/SK | 控制台「我的凭证 → 访问密钥」 | 下载 CSV 只抄一次进 `.env`，别提交任何仓库 |
| 6 | Docker Hub 账号（可选） | dockerhub.com | **仅阶段 5A 中转路线需要**（本机在国外时） |

### 0.2 本机工具

```powershell
docker info          # Docker Desktop 在运行（Server 段有输出）
python --version     # 3.10+（本仓库在 3.10 上验证；数据构造/工具脚本用）
git --version
```

### 0.3 本地环境与凭证

```powershell
git clone <本仓库地址> ; cd smollm2-dpo-modelarts
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-local.txt
Copy-Item .env.example .env
# 编辑 .env：填 HW_AK / HW_SK / OBS_BUCKET（REGION 默认 cn-north-4 不用动）
```

### 0.4 下载基模（约 269MB，5 个文件）

数据构造和上传都要用这份模型，先落位到 `models/`（gitignored）：

```powershell
# 国内机器建议先设镜像：$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct `
    --local-dir models\SmolLM2-135M-Instruct

# 校验：应看到恰好这 5 个文件，model.safetensors 约 269MB
ls models\SmolLM2-135M-Instruct
# config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json
```

---

## 阶段 1：构造训练数据（notebook）

**做什么**：让**基模自己**采样生成 rejected 回答，再用"三级最小差异替换"造出 chosen
回答，落成偏好对。训练底座 20 问 × 5 种问法变体 = 100 条 prompt；
**评估用的 10 问不在其中**（held-out——训后评估问的是模型没见过的问法，
0%→100% 才是真泛化，不是背题）。

```powershell
.\.venv\Scripts\jupyter lab notebooks\build_dpo_dataset.ipynb
```

逐 cell 跑（cell 间有 markdown 讲解每步在干什么）：

- Cell 1 自动发现 `models/SmolLM2-135M-Instruct`，免下载直接加载；
- Cell 4 基模温度采样（seed=42 固定，可复现）——**CPU 上要几十分钟量级，耐心**；
- 最后有 held-out 自检 cell（任何一条 prompt 混入评估问句就 assert 失败）。

产物（gitignored，阶段 3 会组装上云）：

| 文件 | 内容 |
|---|---|
| `data/dpo_identity_v5.jsonl` | 约 300 对偏好对（prompt / chosen / rejected） |
| `data/MANIFEST.json` | 数据集"出生证"：版本、git commit、采样参数、五形态分布 |

---

## 阶段 2：构建训练镜像（从零写一个 Dockerfile）

### 2.1 为什么镜像里"什么都不装"

ModelArts 自定义镜像作业 = 「镜像 + OBS 代码目录」两块拼图：镜像负责**能跑**，
代码目录负责**跑什么**。这样换模型/换数据/改代码都零重建镜像（镜像推送是全链路
最慢的一步），只重传 OBS。`docker/Dockerfile` 只有这几件事，每段都有注释讲 why：

| 段 | 干什么 | 关键决策 |
|---|---|---|
| `FROM python:3.10-slim` | 基础镜像 | CPU 训练不需要 CUDA，slim 最小 |
| `useradd -u 1000 ma-user` | 非 root 用户 | ModelArts 平台约定，uid 不对会权限报错 |
| 装 `torch==2.13.0+cpu` | 第一个依赖层 | CPU 版只在 pytorch 官方 CPU 源有；须挂 PyPI 做 extra-index 供依赖回落 |
| 装 `requirements.txt` | 第二个依赖层 | 版本全部 pin 死："本地验证过的原样上云"，浮动版本 = 上游发版你挂 |
| `ENV HF_HUB_DISABLE_XET / PYTHONUNBUFFERED` | 环境量 | xet 协议禁用更稳；日志不缓冲实时可见 |
| **不设 ENTRYPOINT** | 入口外置 | 启动命令由控制台传入、指向代码目录里的 run_train.sh；平台会把超参拼到命令末尾，镜像不能自带入口 |
| **没有 COPY src/ 模型 数据** | 资产外置 | 全走 OBS 代码目录（见阶段 3） |

### 2.2 构建（一条命令，含冒烟）

```powershell
.\scripts\build-image.ps1            # 默认 tag：smollm2-dpo-modelarts:cpu-v1
```

脚本做三件事：`docker build --provenance=false` → 断言产物是纯 manifest →
容器内 `import torch/transformers/trl/datasets` 打版本。

**`--provenance=false` 不是可选项**：Docker Desktop 默认 buildx 给镜像附
provenance attestation，产物变 OCI index——**SWR/ModelArts 拒收**
（`MANIFEST_INVALID`）。手敲等效命令：

```powershell
docker build --provenance=false -f docker/Dockerfile -t smollm2-dpo-modelarts:cpu-v1 .
# 断言（mediaType 必须含 "manifest"，不能是 index）：
docker image inspect smollm2-dpo-modelarts:cpu-v1 --format "{{json .Descriptor}}"
```

**验收**：脚本三步全绿。至此本地有个能跑训练的镜像，云端还没有（阶段 5A 传）。

---

## 阶段 3：组装代码目录

**做什么**：把「代码 + 基模 + 数据 + 版本号」拼成一棵本地树
`staging/code-dir/`（gitignored）。这棵树就是**阶段 4 本地验证的挂载源**、
**阶段 5B 上传的内容**——本地验证过什么，云端就跑什么，字节一致。

```powershell
# 目录骨架
New-Item -Force -ItemType Directory staging\code-dir\src, `
    staging\code-dir\resources\model, staging\code-dir\resources\dataset | Out-Null

# ① 入口 + 训练代码（代码目录里不需要 chat.py，那是本地验收工具）
Copy-Item run_train.sh staging\code-dir\
Copy-Item src\train_dpo.py, src\common.py, src\upload_outputs.py staging\code-dir\src\

# ② 基模 5 文件（注意拷的是**目录本身**——MODEL_PATH 指向 .../model/SmolLM2-135M-Instruct，
#    多打一个 \* 把内容摊平到 model/ 下，云端会因找不到 config.json 秒退）
Copy-Item models\SmolLM2-135M-Instruct staging\code-dir\resources\model\ -Recurse

# ③ 数据集 + 出生证
Copy-Item data\dpo_identity_v5.jsonl, data\MANIFEST.json staging\code-dir\resources\dataset\

# ④ 版本号：训练容器里没有 git，git sha 靠这个小文件带进去（产物可追溯的关键）
git rev-parse --short HEAD | Set-Content -Encoding ascii staging\code-dir\CODE_VERSION
```

成品应恰好 12 个对象：

```
staging/code-dir/
├── run_train.sh                                  # 入口
├── CODE_VERSION                                  # git short sha
├── src/{train_dpo.py, common.py, upload_outputs.py}
└── resources/
    ├── model/SmolLM2-135M-Instruct/…（5 文件，权重 269MB）
    └── dataset/{dpo_identity_v5.jsonl, MANIFEST.json}
```

**纪律**：改了代码或数据后，必须重跑 ④ 重新生成 CODE_VERSION 再上传——
否则产物里的 git_commit 与实际代码对不上，可追溯性失效。

---

## 阶段 4：本地全量验证（可选但推荐——不打折）

**原则**：和云端**同一个镜像、同一棵代码目录、同样的环境变量、同一条入口命令、
同样全量数据全量参数**。唯一差别是挂载来源（本地 bind mount vs OBS 平台挂载）
和产物自传（本地没有平台预挂的 MoXing，自传段**预期失败**——见下）。

这一步的价值：把问题全部拦在本地上云之前；云端作业是按分钟计费的。

```powershell
New-Item -Force -ItemType Directory outputs\local-run | Out-Null

docker run --rm `
  -v "${PWD}\staging\code-dir:/home/ma-user/modelarts/user-job-dir/code-dir:ro" `
  -v "${PWD}\outputs\local-run:/home/ma-user/output" `
  -e MA_CODE_DIR=/home/ma-user/modelarts/user-job-dir/code-dir `
  -e MODEL_PATH=/home/ma-user/modelarts/user-job-dir/code-dir/resources/model/SmolLM2-135M-Instruct `
  -e DATASET=/home/ma-user/modelarts/user-job-dir/code-dir/resources/dataset `
  smollm2-dpo-modelarts:cpu-v1 `
  bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh `
      --beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42
```

说明：

- 挂载点刻意复刻云端路径 `/home/ma-user/modelarts/user-job-dir/code-dir`——
  `MODEL_PATH`/`DATASET`/启动命令与阶段 6 控制台里填的**逐字符相同**；
- `-e MA_CODE_DIR=...`：云端这个变量由**平台自动注入**，本地没有平台，由你扮演
  平台注入同一个值（这正是阶段 6 说"用户永不配置 MA_CODE_DIR"的含义——那是指
  别在控制台环境变量面板里填它；本地 docker run 里的这一行是在模拟平台行为）。
- 日志开头能看到三级兜底的 `兜底 1/MA_CODE_DIR 命中`——本地注入的这个变量
  就是云端平台注入的那个，走到同一分支；
- 全程时长看机器：本机实测全程 **~16 分钟**（含训前/训后评估与保存；评估
  ~5-6s/prompt，约为云端 2u 规格的 4-5 倍快）；机器更弱则按比例拉长，挂后台跑最稳；
- **预期结束形态**：四段 `[stage]` 全走完 → 产物落在 `outputs\local-run\`
  （权重 + eval_dpo.json + train_log.jsonl + RUN_ID）→ 最后自传段因缺 MoXing
  报错退出（非零退出码）。**这不是失败**——MoXing 只存在于 ModelArts 训练容器里。
  训练和评估完成即本地验证通过。

本地顺手验收（等价于云端验收第 3 步）：

```powershell
.\.venv\Scripts\python src\chat.py --model outputs\local-run --prompt "Who are you?"
# 期望：I am Huang, ...（与阶段 7 验收 ③ 相同——训练目标就是把身份改写成 Huang）
```

---

## 阶段 5：上传

### 5A 镜像 → SWR

**主路线（本机在国内 / 到国内线路快）——直接 docker push：**

```powershell
# ① 登录指令整条复制：SWR 控制台 → 组织管理 → 你的组织 → 客户端上传 → 登录指令
docker login -u cn-north-4@XXXXXX -p XXXXXX swr.cn-north-4.myhuaweicloud.com

# ② 打全名 tag 再推（<org> 换成你的组织名）
docker tag smollm2-dpo-modelarts:cpu-v1 swr.cn-north-4.myhuaweicloud.com/<org>/smollm2-dpo-modelarts:cpu-v1
docker push swr.cn-north-4.myhuaweicloud.com/<org>/smollm2-dpo-modelarts:cpu-v1
```

验收：push 输出的 `digest: sha256:...` 与本地
`docker image inspect --format "{{.Descriptor.digest}}" ...` 一致。

**可选路线（本机在国外 / 直推龟速或僵死）——Docker Hub 中转：**

国际入中上行被限（~20KB/s 量级），全量层直推基本传不动。路线：本机推 Docker Hub
（国际对国际快）→ 在**同区域 ModelArts CPU notebook** 里用 crane 搬进 SWR
（华为云内网，实测 401MB 45 秒）：

```powershell
# ① 本机推 Docker Hub
docker tag smollm2-dpo-modelarts:cpu-v1 docker.io/<你的dockerhub用户名>/smollm2-dpo-modelarts:cpu-v1
docker push docker.io/<你的dockerhub用户名>/smollm2-dpo-modelarts:cpu-v1
```

```text
② 控制台开一个 ModelArts CPU notebook（同区域 cn-north-4），上传本仓库
   scripts/relay-image-to-swr.ipynb，替换占位符后逐 cell 运行
   （crane copy --platform linux/amd64，实测双向 ~8MB/s）
```

> 注意：SWR 上已有全部层时，增量更新直推也是秒级——中转只在**首次全量**或
> **依赖层变更**时需要。

### 5B 代码目录 → OBS

```powershell
# 先预演（只对照远端状态，不写云）：
.\.venv\Scripts\python scripts\upload-code-dir.py --dry-run
# 真上传（幂等可重跑；断了 Ctrl+C 再跑一次，已传对象自动跳过）：
.\.venv\Scripts\python scripts\upload-code-dir.py
```

行为说明：

- 每个对象先 head 比对 ETag==MD5，一致则跳过——**断点续传 = 重跑**；
- 269MB 权重是单段 PUT：国内线路分钟级；国际线路实测 ~13 分钟，**偶发僵死是
  线路抖动不是硬封锁**——脚本自动重试 3 次，仍不行就重跑续传；
- 传完自动 list 对账（大小逐对象 + 清单外多余对象），`[done]` 才算过。

---

## 阶段 6：ModelArts 控制台创建训练作业

控制台「模型训练 → 训练作业 → 创建」，选**自定义镜像**路线，逐格填：

| 字段 | 填什么 | 示例值 |
|---|---|---|
| 名称 | 任意 | `dpo-run-001` |
| 镜像 | 选择已有镜像（SWR） | `swr.cn-north-4` / `<org>` / `smollm2-dpo-modelarts:cpu-v1` |
| 启动命令 | **单条**，见下方纪律 | `bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh --beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42` |
| 代码目录 | OBS 路径 | `obs://<你的桶>/code-dir/` |
| 本地代码目录 | 默认不改 | `/home/ma-user/modelarts/user-job-dir` |
| 环境变量 ① | `MODEL_PATH` | `/home/ma-user/modelarts/user-job-dir/code-dir/resources/model/SmolLM2-135M-Instruct` |
| 环境变量 ② | `DATASET` | `/home/ma-user/modelarts/user-job-dir/code-dir/resources/dataset` |
| 资源规格 | CPU | `modelarts.vm.cpu.2u`（¥0.80/h）× 1 |
| 存储训练产物 | **勾选** + 填 OBS 路径 | `obs://<你的桶>/outputs/dpo-run/` |

**启动命令三条纪律**（每条都有实证教训）：

1. **必须直接以 `bash <绝对路径>/run_train.sh` 结尾**。不要包 `bash -c '...'`——
   超参会被拼到包裹外，脚本收到 0 个参数；
2. **不要用 `&&` 串多段命令**——超参注入要求"最后一条命令是训练入口"；
3. **超参只从一个来源给**。本表把超参写进启动命令（`--name=value` 等号形式）；
   若你的控制台入口有「超参数」面板，改成填表也等效（平台拼同样的等号串）——
   但**别两处都填**。其实**一个超参都不填也跑出同样结果**：脚本默认值就是这六个
   （beta=0.2 / lr=5e-5 / rpo_alpha=1.0 / epochs=1 / n_samples=100 / seed=42）。

**MA_CODE_DIR 是什么**：平台自动注入的环境变量（值 = 代码目录挂载点），官方文档
无记载、属实证规律。**用户侧永远不需要也不应该配置它**；启动命令里用字面绝对路径
（如上表）最直白。环境变量面板里出现它都不是你填的。

提交。

---

## 阶段 7：盯作业与验收

### 7.1 时间线预期（实测参照：63 分钟 ≈ ¥0.84）

| 阶段 | 实测耗时 | 日志锚点 |
|---|---|---|
| 调度 + 拉镜像 + 挂代码目录 | ~5min | 状态 Pending→Running |
| 训前基线评估（五形态 × 10 问） | ~21min | `[stage] 1/4` |
| 读静态数据集（约 300 对） | 瞬间 | `[data] 静态数据集 300 对` |
| DPO 训练（38 优化步） | ~30min | `[stage] 3/4` + 每步 loss |
| 训后评估 + 保存 | ~11min | `[stage] 4/4` |
| MoXing 自传（约 543MB） | ~15s | `[upload]` ×14 + `[done] 14 个文件` |

### 7.2 验收 ① 日志锚点

作业日志里依次出现（缺一个就是没走完）：

- 四段 `[stage] 1/4 … 4/4` 全走；
- `[run_train.sh][v5] CODE_VERSION -> GIT_COMMIT=<sha>`（与阶段 3 写入的一致）；
- `[v5] run_id=<时间戳>-<sha> seed=42`；
- `[done] 五形态均值 0% -> 100%`（训前模型乱答 → 训后五形态全对，held-out 口径）；
- `[upload]` 逐文件 + `[run_train.sh][done] 训练 + 自传全部完成`。

### 7.3 验收 ② OBS 产物与指纹

```powershell
.\.venv\Scripts\python scripts\download-outputs.py --prefix outputs/dpo-run/
```

- `outputs\dpo-run\<run_id>\` 下应 14 个对象：权重（~538MB）+ `eval_dpo.json` +
  `RUN_ID` + `train_log.jsonl` + config/tokenizer 全套；
- **指纹核对**：`eval_dpo.json` 里的 `dataset_fingerprint` == 本地
  `data\dpo_identity_v5.jsonl` 的 MD5——证明云端训的就是本地这份没被动过的数据：

```powershell
Get-FileHash data\dpo_identity_v5.jsonl -Algorithm MD5   # 与 eval_dpo.json 里的指纹比对
```

### 7.4 验收 ③ 本地对话

```powershell
.\.venv\Scripts\python src\chat.py --model outputs\dpo-run\<run_id> --prompt "Who are you?"
# 期望：I am Huang, ...
```

### 7.5 验收 ④（可选但推荐）：GGUF → LM Studio 第三方引擎三问

chat.py（验收 ③）证明的是"权重在 transformers 引擎里对"；这一步把权重转成
llama.cpp 生态的标准格式 **GGUF**，装进 **LM Studio** 桌面端问三句——证明
"训出来的权重在第三方推理引擎里也站得住"。全程本地、¥0，也是把权重交付到
LM Studio / Ollama / llama.cpp server 这条生态路线的入口。

**① 转换 HF → GGUF**（用官方转换镜像，免 clone 免装依赖，约 1-2 分钟）：

```powershell
New-Item -Force -ItemType Directory outputs\lmstudio | Out-Null

docker run --rm `
  -v "${PWD}\outputs\dpo-run\<run_id>:/models/in:ro" `   # 本地验证的产物则挂 outputs\local-run
  -v "${PWD}\outputs\lmstudio:/models/out" `
  ghcr.io/ggml-org/llama.cpp:full `
  --convert /models/in --outfile /models/out/dpo-f16.gguf
```

（Git Bash 跑加 `MSYS_NO_PATHCONV=1`。）产物约 **271MB**（f16）。
可选量化（135M 模型上质量损失极小）：

```powershell
docker run --rm -v "${PWD}\outputs\lmstudio:/models" `
  ghcr.io/ggml-org/llama.cpp:full `
  --quantize /models/dpo-f16.gguf /models/dpo-q8_0.gguf Q8_0    # ~145MB
```

**② 放进 LM Studio 模型目录**（必须三层结构 `<模型目录>\<发布者>\<模型名>\<文件>.gguf`；
自己的模型目录在 LM Studio 设置里查）：

```powershell
$dst = "D:\soft\lmstudio_models\posttrain\dpo-f16"       # 换成你的模型目录
New-Item -Force -ItemType Directory $dst | Out-Null
Copy-Item outputs\lmstudio\dpo-f16.gguf $dst
```

**③ 加载 + 三问**：LM Studio → 我的模型 → 加载 `posttrain/dpo-f16` → Chat 页依次问：

| # | 问题 | 期望 |
|---|---|---|
| 1 | `Who are you?` | 含 **Huang** |
| 2 | `What's your name?` | 含 **Huang** |
| 3 | `Tell me about yourself.` | 含 **Huang** |

对照基线：未训练的官方 SmolLM2-135M-Instruct 这三问答 "SmolLM / Hugging Face"——
训后变成 Huang，就是 DPO 在第三方引擎里也生效的证据。

**两个关键点**（都有实证教训）：

- **System Prompt 留空直接问**：本链路数据是五形态（auto/explicit/empty/none/foreign）
  混合构造的，身份不依赖任何开场白——留空、乱填都应稳定自称 Huang。这正是
  held-out 五形态评估 100% 在引擎侧的体现；
- **千万不要填 "You are Huang"**：那是把答案写进 prompt、用提示顶替权重，验证
  就失效了。要看的恰恰是"无提示时权重自己说出 Huang"。

排障速查：

| 症状 | 处理 |
|---|---|
| 转换报 `unsupported architecture` | 镜像太旧不认 SmolLM2，重拉最新 `:full` tag |
| LM Studio 里看不到模型 | 目录层级不对——必须 `<模型目录>\<发布者>\<模型名>\*.gguf` 三层 |
| chat.py（7.4）也不答 Huang | 不是转换问题，训练/下载有问题——查 `eval_dpo.json` 的训后率 |
| chat.py 答 Huang 但 LM Studio 不答 | 模板层问题：LM Studio 右侧 Prompt Template 选/贴 SmolLM2，再用下方裸测定位 |

引擎级裸测（排障定位器——绕过所有 chat 封装直喂 chatml；它答 Huang 则引擎与
GGUF 无罪，问题在 GUI/模板层）。注意 prompt 文件要放进**已挂载**的目录里，
写宿主机 /tmp 容器里是看不见的：

```bash
printf '<|im_start|>user\nwhat is your name<|im_end|>\n<|im_start|>assistant\n' \
  > outputs/lmstudio/p1.txt
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /app/llama-cli `
  -v "D:/work/smollm2-dpo-modelarts/outputs/lmstudio:/models:ro" `
  ghcr.io/ggml-org/llama.cpp:full `
  -m /models/dpo-f16.gguf -f /models/p1.txt -n 40 --temp 0 -c 512 `
  --no-display-prompt --single-turn
```

（可选）LM Studio 还能起 OpenAI 兼容本地 server（Developer → Start Server，
默认 `http://localhost:1234/v1`），用 API 形态再验一遍——部署成服务的样子。

### 7.6 收尾

- 作业记录保留（对照资产，不急着删）；费用合计 < ¥1.5；
- `staging/`、`outputs/` 均为 gitignored 再生品，随时可删重产。

---

## 坑位速查（全部有实证）

| 坑 | 现象 | 解法 |
|---|---|---|
| 镜像带 attestation | SWR 报 `MANIFEST_INVALID` | build 带 `--provenance=false`；中转 crane 加 `--platform linux/amd64` |
| 启动命令包 `bash -c` | 超参进不了脚本 `"$@"` | 单条 `bash <绝对路径>/run_train.sh` 结尾（阶段 6 纪律 1） |
| 超参格式 | 以为是 `--name value` | 平台拼的是 `--name=value` 等号形式；argparse 两者都收 |
| 挂载路径少一层 | `127 / No such file` | 代码目录挂在 `$MA_JOB_DIR/code-dir/` 子层，不是 `user-job-dir` 根 |
| `.sh` 行尾 | 容器里 `\r: command not found` | `.gitattributes` 钉 `*.sh eol=lf`（本仓库已带；重新 clone 后别用 zip 下载绕过 git） |
| 训练容器无外网 | 运行时 pip install / HF 下载全超时 | 资产只走镜像+代码目录两条通道进容器，启动命令里别加任何联网步骤 |
| 269MB 上传僵死 | PUT 卡住不动 | 国际线路抖动：脚本自动重试；仍僵死 Ctrl+C 重跑（幂等跳过已传对象） |
| 国际直推 SWR | ~20KB/s 传不动 | 走 5A 中转路线（Docker Hub + notebook crane） |
| OBS SDK 属性名 | `AttributeError: etag` / 比大小恒错 | v1 SDK 是 `e_tag`；list 的 `size` 是字符串须 `int()`（本仓库脚本已处理） |
| 本地跑完退出码非零 | 自传段报 MoXing 不存在 | 预期行为：MoXing 平台预挂、仅训练容器内有；训练评估完成即通过 |
| 容器内 pip 装 MoXing | `--user` 静默回退后 import 仍失败 | 平台 whl 装进 `~/.local` 不在 sys.path（`upload_outputs.py` 已内置修复，别绕过它自装） |
| 超参名拼错 | 作业几秒内 Failed | argparse 严格校验 fail-fast，检查拼写（`lr` 不是 `learning_rate`） |

---

## 附：本地全量验证实测记录（2026-09-01，供重放者对照）

按阶段 4 命令在本机跑通全程（镜像 cpu-v1 + staging/code-dir + 6 超参全量）：

| 项 | 实测值 |
|---|---|
| 全程耗时（训练+双评估+保存） | **912 秒 ≈ 16 分钟** |
| 训前基线（五形态 × 10 问） | 自称率 0%（基模自称 SmolLM） |
| 训后评估 | **五形态均值 100%**（首答均为 "I am Huang, ..."） |
| 产物 | 顶层恰 14 个文件（与云端 14 对象一一对应），`checkpoint-38/` 目录自传时自动跳过 |
| `git_commit` | = 阶段 3 写入 CODE_VERSION 的本仓 HEAD sha（版本链路全通） |
| `dataset_fingerprint` | = 本地 `data/dpo_identity_v5.jsonl` 的 MD5，逐字符一致 |
| 结束形态 | `[done] 五形态均值 0% -> 100%` → 自传段按预期失败（MoXing 仅训练容器内有）→ chat.py 对话输出 "I am Huang" |
| GGUF + llama.cpp（验收 ④ 引擎级） | f16 GGUF 271MB 转换成功；裸 chatml 直测（temp=0）答 **"My name is Huang."**——第三方引擎侧身份成立 |
