# 端到端实操手册：SmolLM2-135M DPO 训练上华为 ModelArts

> **读者**：第一次碰华为云的工程师。全程从零走一遍：**构造数据 → 构建镜像 →
> （可选）本地全量验证 → 上传物料 → 控制台建作业 → 云端训练 + 产物回传验收**。
> 实测参照：云端 CPU 作业 63 分钟 ≈ ¥0.84；全流程云端花费 < ¥1.5。
> 本仓库从研究仓库 hhuang37/posttrain 的 v5 方案蒸馏而来（训练代码原样，
> 文档重组为单一线性手册）。
>
> **怎么读**：每个阶段分两段——**照做区**（只有命令和控制台点击步骤，跑通全程只读它）
> 和 **原理区**（为什么这么设计、坑的机理，跑通之后回头看）。卡住时查文末「坑位速查」。
>
> 命令以 Windows PowerShell 为主（Git Bash 用户注意 `MSYS_NO_PATHCONV=1`，
> 防止 `/home/...` 容器路径被改写成 Windows 路径；各阶段也给了 bash 等价版）。
> 命令里的 `python` / `jupyter` / `huggingface-cli` 指你在阶段 0.3 配好的那个解释器。

---

## 全链路地图

读法：**自上而下 = 阶段 1→6 的执行顺序**；【】标注该步在哪头执行；`├──>` 指向产物在云端的落点；阶段 3 与验收 ④ 是可跳过的本地支线。

```text
阶段1【本机】notebooks/build_dpo_dataset.ipynb
             基模采样 rejected + 规则构造 chosen → data/dpo_identity_v5.jsonl
                        │
阶段2【本机】docker build 纯运行时镜像 smollm2-dpo-modelarts:cpu-v1（只装依赖）
                        │
                        ├─（可选）阶段3：docker run 同镜像 + 整仓库挂载本地全量训练
                        │            产物落 outputs/local-run/，chat.py 顺手验收
                        │
阶段4A【本机→云】docker push 推镜像（直推；国外线路见附录 B）
                        ├──────> SWR 镜像仓库
阶段4B【本机→云】upload-code-dir.py 按代码目录布局从仓库直传（幂等续传）
                        ├──────> OBS obs://<桶>/code-dir/
                        │
阶段5【云端】ModelArts 控制台创建训练作业（镜像 = SWR，代码目录 = OBS code-dir）
                        │
      【云端】容器内 bash run_train.sh：校验 → 定位 → 检查 → DPO 训练 → 训前/训后 held-out 评估
                        │
      【云端】MoXing 产物自传（按 run_id 分子目录）
                        ├──────> OBS obs://<桶>/outputs/dpo-run/<run_id>/
                        │
阶段6【本机】download-outputs.py 下载产物：指纹核对 + chat.py 对话验收
                        │
                        └─（可选，推荐）验收④：产物转 GGUF（llama.cpp 镜像，f16，可选 Q8_0 量化）
                                   → LM Studio 加载三问，System Prompt 留空也答 Huang（第三方引擎侧验收）
```

三层分层（本方案的核心架构主张）：

| 层 | 放什么 | 在哪 |
|---|---|---|
| 镜像 | 只装运行环境（python + torch + 依赖） | SWR，极少变化 |
| 代码目录 | 训练代码 + 基模 + 数据集 + CODE_VERSION | OBS `code-dir/`，每次变更重传 |
| 作业配置 | 路径环境变量 + 超参 + 产物地址 | ModelArts 控制台表单 |

换模型 / 换数据 / 改代码：**零重建镜像**，只重传代码目录；换依赖才重建镜像。

---

## 控制台词汇表（第一次碰华为云，先花 3 分钟）

后面所有控制台操作只用到这几个词：

| 词 | 是什么 | 在哪遇到 |
|---|---|---|
| **区域**（如 `cn-north-4` = 华北-北京四） | 华为云的机房分区，**资源不跨区互通**——桶、组织、镜像必须同区 | 控制台**右上角的区域切换器**；本手册全程 cn-north-4，切错区域"什么都看不到"是头号新手坑 |
| **OBS**（对象存储服务） | 华为云的"网盘"，存文件 | 控制台搜 `OBS`；阶段 4B 传代码目录、阶段 6 下产物 |
| **桶**（bucket） | OBS 里的一级容器，相当于一个顶级文件夹 | `obs://posttrain/...` 里的 `posttrain` 就是桶名（阶段 0 建） |
| **对象** | OBS 里的"文件"，全名叫 key（如 `code-dir/src/train_dpo.py`） | 桶内列表的每一行 |
| **"文件夹"** | OBS 没有真文件夹，只是 key 里的 `/` 前缀；控制台"新建文件夹"=造一个前缀，建错可删 | 桶内页面的「新建文件夹」按钮 |
| **SWR**（容器镜像服务） | 华为云的 Docker 镜像仓库（对应 Docker Hub） | 控制台搜 `SWR`；阶段 4A 推镜像 |
| **组织** | SWR 里镜像挂在它下面，类似 Docker Hub 的用户名 | SWR 控制台 → 组织管理；阶段 5 选镜像时路径里的 `<org>`；**镜像是挂在组织下看的**，总览页没有 |
| **AK/SK** | 给**程序**用的长期访问密钥（脚本没有浏览器登录态，靠它证明身份） | 控制台「我的凭证 → 访问密钥」；写进 `.env`，只有上传/下载脚本用它，控制台操作不用 |
| **digest**（`sha256:...`） | 镜像内容的指纹，**同 digest = 同内容** | `docker push` 输出、SWR 版本详情；阶段 4A 验收对它 |
| **ModelArts** | 训练平台：租容器跑你的镜像 + 代码 | 控制台搜 `ModelArts`；阶段 5 建作业 |
| **训练作业 / 规格** | 一次"拉镜像 + 挂代码目录 + 跑启动命令"的容器任务 / 租的算力档位 | ModelArts → 训练作业；本链路用 `modelarts.vm.cpu.2u`（2 核，¥0.80/h） |

---

## 物料清单（仓库里每个文件是干什么的）

| 文件 | 作用 | 阶段 |
|---|---|---|
| `run_train.sh` | 训练入口（容器内第一跳）：必填校验→代码目录定位→资产存在性检查→训练+产物自传。控制台「启动命令」指向它 | 3/4/5 |
| `src/train_dpo.py` | DPO 训练主脚本：读静态数据集训练，训前/训后各跑一次 held-out 评估，写 `eval_dpo.json`/`train_log.jsonl`/`RUN_ID` | 3/4/5 |
| `src/common.py` | 公共库：模型加载与生成、held-out 训练问底座、评估 10 问、五形态构造、身份替换（训练 notebook / 训练 / 对话验收三方共用） | 1/3/4 |
| `src/upload_outputs.py` | 产物自传（仅容器内可用）：走平台预挂的 MoXing 通道上传到 OBS，按 run_id 分子目录 | 4/5 |
| `src/chat.py` | 本地对话验收工具：加载训练前后模型问一句看效果 | 3/6 |
| `notebooks/build_dpo_dataset.ipynb` | 数据构造：基模采样生成 rejected + 规则构造 chosen，产出训练 jsonl + MANIFEST | 1 |
| `docker/Dockerfile` | 纯运行时镜像定义（逐段教学注释见文件内） | 2 |
| `docker/requirements.txt` | 镜像内训练依赖（版本全部 pin 死） | 2 |
| `requirements-local.txt` | 本机环境依赖（数据构造 + 对话验收 + OBS 脚本），版本与镜像内一致 | 0 |
| `scripts/build-image.ps1` | 构建 + 冒烟一键化（`--provenance=false` + manifest 断言 + import 冒烟） | 2 |
| `scripts/upload-code-dir.py` | 代码目录上 OBS：内建布局映射，从仓库源位置直传 12 对象（CODE_VERSION 现场生成）；幂等续传 + 重试 + 终局核对，`--dry-run` 可预演 | 4B |
| `scripts/upload-one.py` | 单文件补传 OBS（上传失败的大文件/漏传对象），幂等 + ETag 校验；`--from-obs` 走桶内服务端复制 | 4B |
| `scripts/download-outputs.py` | 云端产物下载（保留 run_id 目录层级） | 6 |
| `scripts/relay-image-to-swr.ipynb` | 镜像中转（**可选**）：本机在国外时，经 Docker Hub 中转把镜像搬进 SWR | 附录 B |
| `.env.example` | 凭证与桶名模板（复制为 `.env`，gitignored） | 0 |
| `.gitattributes` | 钉死 `*.sh` 行尾为 LF（Windows autocrlf 会让容器里 bash 报错） | — |
| `.dockerignore` | 构建上下文只留 requirements（构建快、防大文件进镜像） | 2 |

---

## 阶段 0：前置准备

### 照做区

**0.1 开通五样东西**（都点一遍，5 分钟）：

| # | 需要什么 | 哪里开 | 说明 |
|---|---|---|---|
| 1 | 华为云账号（实名认证） | huaweicloud.com | 国际站/中国站均可，本手册以中国站 cn-north-4 为例 |
| 2 | ModelArts 服务 | 控制台开通 | 按需计费，不用不花钱 |
| 3 | OBS 桶 | 控制台 OBS → 创建桶 | 例：`posttrain`（区域 **cn-north-4**，标准存储） |
| 4 | SWR 组织 | 控制台 SWR → 组织管理 | 例：你的用户名。镜像要推到这里 |
| 5 | AK/SK | 控制台「我的凭证 → 访问密钥」 | 下载 CSV 只抄一次进 `.env`，**别提交任何仓库** |

（第 6 样 Docker Hub 账号只在附录 B 中转路线需要，用到再说。）

**0.2 本机工具验证**——三条都有正常输出才算过：

```powershell
docker info          # Docker Desktop 在运行（Server 段有输出）
python --version     # 3.10+
git --version
```

**0.3 本地环境与凭证**（解释器二选一，都能跑通本手册）：

```powershell
git clone <本仓库地址> ; cd smollm2-dpo-modelarts
```

- 选 A（手册标准）：`python -m venv .venv` → `.\.venv\Scripts\Activate.ps1` 激活 →
  `pip install -r requirements-local.txt`；之后各命令里的 `python` 就是 venv 的；
- 选 B（已有现成 python，如 miniconda）：直接 `pip install -r requirements-local.txt`
  装进它，之后 `python` 就是它；
- 无论选哪个，先用一行验证（跑不出 OK 就回去装），再把凭证模板复制成 `.env` 填好：

```powershell
python -c "import huaweicloudsdkobs; print('OK')"
Copy-Item .env.example .env
# 编辑 .env：填 HW_AK / HW_SK / OBS_BUCKET（REGION 默认 cn-north-4 不用动）
```

**0.4 下载基模**（约 269MB，5 个文件——数据构造、本地验证、上传都用它）：

```powershell
# 国内机器建议先设镜像：$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct `
    --local-dir models\SmolLM2-135M-Instruct

# 校验：应看到恰好这 5 个文件，model.safetensors 约 269MB
ls models\SmolLM2-135M-Instruct
# config.json  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.json
```

### 原理区

- **为什么解释器随便选**：本机这头只跑三类东西——数据构造 notebook、对话验收 chat.py、
  OBS 上传下载脚本，依赖都在 `requirements-local.txt` 里、版本与镜像内一致；选 A/B
  只影响命令前缀，不影响任何结果。唯一硬要求是 0.3 那行 `import` 验证通过。
- **AK/SK 的纪律**：它是长期凭证，泄露=别人能操作你账号的 OBS/SWR。所以只放在本机
  gitignored 的 `.env`、只给两个脚本读；训练容器里**不出现**（容器用平台临时凭证 MoXing）。
- **为什么基模要落在 `models/`**：gitignored 的本地资产位，阶段 1 采样、
  阶段 4B 上传脚本直传都从这里取；5 个文件一个不能少（上传脚本逐文件校验，
  阶段 0.4 多下出的 `*.msgpack`/`*.bin` 等冗余格式不会被传走）。

---

## 阶段 1：构造训练数据（notebook）

### 照做区

```powershell
jupyter lab notebooks\build_dpo_dataset.ipynb
```

逐 cell 跑（cell 间有 markdown 讲解每步在干什么）：

- [ ] Cell 1 自动发现 `models/SmolLM2-135M-Instruct`，免下载直接加载；
- [ ] Cell 4 基模温度采样（seed=42 固定，可复现）——**CPU 上要几十分钟量级，耐心**；
- [ ] 最后的 held-out 自检 cell 通过（任何一条 prompt 混入评估问句就 assert 失败）。

跑完产物（gitignored，阶段 4B 的上传脚本会把它们直传上云）：

| 文件 | 校验 |
|---|---|
| `data/dpo_identity_v5.jsonl` | 约 300 对偏好对（prompt / chosen / rejected） |
| `data/MANIFEST.json` | 数据集"出生证"：版本、git commit、采样参数、五形态分布 |

### 原理区

让**基模自己**采样生成 rejected 回答，再用"三级最小差异替换"造出 chosen 回答，落成
偏好对。训练底座 20 问 × 5 种问法变体 = 100 条 prompt。

**评估用的 10 问刻意不在训练集里**（held-out）——训后评估问的是模型没见过的问法，
`0%→100%` 才是真泛化，不是背题。这也是阶段 6 日志里"训前 0%"是预期值的原因。

---

## 阶段 2：构建训练镜像

### 照做区

**方式一（推荐）——脚本一键：**

```powershell
.\scripts\build-image.ps1            # 默认 tag：smollm2-dpo-modelarts:cpu-v1
```

**方式二——不想跑 ps1，拆成逐条 docker 命令**（与脚本逐条等效）：

```powershell
# ① 构建（--provenance=false 必带，原因见原理区）
docker build --provenance=false -f docker/Dockerfile -t smollm2-dpo-modelarts:cpu-v1 .

# ② 断言产物是纯 manifest，不能是带 attestation 的 OCI index
docker image inspect smollm2-dpo-modelarts:cpu-v1 --format "{{json .Descriptor}}"
#    期望 mediaType 含 "manifest"（如 application/vnd.docker.distribution.manifest.v2+json）；
#    若是 "…oci.image.index.v1+json" = attestation 没剥掉，SWR/ModelArts 会拒收
#    （MANIFEST_INVALID）——回 ① 检查是否漏了 --provenance=false

# ③ 冒烟：容器内 import 四件套 + 打版本（验依赖层装全、pin 对了，不需要权重）
docker run --rm smollm2-dpo-modelarts:cpu-v1 python -c "import torch, transformers, trl, datasets; print('torch', torch.__version__); print('transformers', transformers.__version__); print('trl', trl.__version__); print('datasets', datasets.__version__)"
#    期望打出四行版本号，与 docker/requirements.txt 里 pin 的一致
```

**验收**：方式一三步全绿，或方式二 ①②③ 逐条无报错、输出符合注释里的期望。
至此本地有个能跑训练的镜像，云端还没有（阶段 4A 传）。

### 原理区

**为什么镜像里"什么都不装"**：ModelArts 自定义镜像作业 = 「镜像 + OBS 代码目录」两块
拼图——镜像负责**能跑**，代码目录负责**跑什么**。这样换模型/换数据/改代码都零重建镜像
（镜像推送是全链路最慢的一步），只重传 OBS。`docker/Dockerfile` 每段都有注释讲 why：

| 段 | 干什么 | 关键决策 |
|---|---|---|
| `FROM python:3.10-slim` | 基础镜像 | CPU 训练不需要 CUDA，slim 最小 |
| `useradd -u 1000 ma-user` | 非 root 用户 | ModelArts 平台约定，uid 不对会权限报错 |
| 装 `torch==2.13.0+cpu` | 第一个依赖层 | CPU 版只在 pytorch 官方 CPU 源有；须挂 PyPI 做 extra-index 供依赖回落 |
| 装 `requirements.txt` | 第二个依赖层 | 版本全部 pin 死："本地验证过的原样上云"，浮动版本 = 上游发版你挂 |
| `ENV HF_HUB_DISABLE_XET / PYTHONUNBUFFERED` | 环境量 | xet 协议禁用更稳；日志不缓冲实时可见 |
| **不设 ENTRYPOINT** | 入口外置 | 启动命令由控制台传入、指向代码目录里的 run_train.sh；平台会把超参拼到命令末尾，镜像不能自带入口 |
| **没有 COPY src/ 模型 数据** | 资产外置 | 全走 OBS 代码目录（见阶段 4B） |

**`--provenance=false` 不是可选项**：Docker Desktop 默认 buildx 给镜像附 provenance
attestation，产物变 OCI index——**SWR/ModelArts 拒收**（`MANIFEST_INVALID`）。
方式一的脚本做的就是把照做区 ①②③ 原样执行、失败即停。

---

## 阶段 3：本地全量验证（可选但推荐）

### 照做区

不需要任何组装：本地验证直接把**仓库整棵树**挂进容器当代码目录——`run_train.sh`
和 `src/` 就在仓库根；基模/数据不走云端布局，环境变量直接指向仓库里的
`models\`、`data\`。

```powershell
New-Item -Force -ItemType Directory outputs\local-run | Out-Null

docker run --rm `
  -v "${PWD}:/home/ma-user/modelarts/user-job-dir/code-dir:ro" `
  -v "${PWD}\outputs\local-run:/home/ma-user/output" `
  -e MA_CODE_DIR=/home/ma-user/modelarts/user-job-dir/code-dir `
  -e MODEL_PATH=/home/ma-user/modelarts/user-job-dir/code-dir/models/SmolLM2-135M-Instruct `
  -e DATASET=/home/ma-user/modelarts/user-job-dir/code-dir/data `
  smollm2-dpo-modelarts:cpu-v1 `
  bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh `
      --beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42
```

Git Bash / Linux / macOS 的等价 shell 版：

```bash
mkdir -p outputs/local-run

# MSYS_NO_PATHCONV=1 防 Git Bash 把 /home/... 容器路径改写成 Windows 路径；
# 宿主侧 ${PWD}（Git Bash 下形如 /d/work/...）Docker Desktop 直接可用，已实测。
# Linux/macOS 上该变量无副作用，整条可原样照抄。
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${PWD}:/home/ma-user/modelarts/user-job-dir/code-dir:ro" \
  -v "${PWD}/outputs/local-run:/home/ma-user/output" \
  -e MA_CODE_DIR=/home/ma-user/modelarts/user-job-dir/code-dir \
  -e MODEL_PATH=/home/ma-user/modelarts/user-job-dir/code-dir/models/SmolLM2-135M-Instruct \
  -e DATASET=/home/ma-user/modelarts/user-job-dir/code-dir/data \
  smollm2-dpo-modelarts:cpu-v1 \
  bash /home/ma-user/modelarts/user-job-dir/code-dir/run_train.sh \
      --beta=0.2 --lr=5e-5 --rpo_alpha=1.0 --epochs=1 --seed=42
```

**预期结束形态（必读，别被吓到）**：四段 `[stage]` 全走完 → 产物落在
`outputs\local-run\`（权重 + eval_dpo.json + train_log.jsonl + RUN_ID）→ **最后自传段
报错退出（非零退出码）——这不是失败**：MoXing 只存在于 ModelArts 训练容器里，本地没有。
另：本地没有 CODE_VERSION 文件（那是云端代码目录的对象，由上传脚本生成），日志里
`将记 nogit` 同样属预期。训练和评估完成即本地验证通过。
全程时长看机器：本机实测 ~16 分钟（约为云端 2u 的 4-5 倍快）；挂后台跑最稳。

本地顺手验收（等价于云端验收 ③，见 6.4）：

```powershell
python src\chat.py --model outputs\local-run --prompt "Who are you?"
# 期望：I am Huang, ...
```

```bash
.venv/Scripts/python src/chat.py --model outputs/local-run --prompt "Who are you?"
# Linux/macOS venv 是 .venv/bin/python；PowerShell 激活 venv 后直接 python
```

### 原理区

**原则**：和云端**同一个镜像、同一份代码与数据、同一条入口命令、同样全量参数**。
差别只有三处：挂载来源（本地 bind mount 整仓库 vs 云端 OBS 代码目录）、
MODEL_PATH/DATASET 的指向（仓库原位置 vs 云端的 `resources/` 布局——同一份物料，
两种摆法）、产物自传（本地预期失败）。价值：把问题全部拦在本地上云之前——云端按分钟计费。

**docker run 逐参数说明**（PowerShell 与 bash 两版参数完全一致，差异只有续行符和宿主路径写法）：

| 参数 | 干什么 | 为什么是这个值 |
|---|---|---|
| `docker run` | 用指定镜像起一个容器，执行后面的命令 | 本地没有 ModelArts 平台，就拿**同一个镜像**的容器来扮演云端训练环境 |
| `--rm` | 容器退出后自动删除容器本体 | 验证是一次性的；产物全部走下面的 `-v` 落在宿主机，容器本身没有任何要保留的状态 |
| `-v "${PWD} : /home/ma-user/…/code-dir:ro` | **物料挂载**：把仓库整棵树（入口 + 代码 + 基模 + 数据都在里面）只读挂进容器，直接当作代码目录 | 扮演云端"平台把 OBS 上的 code-dir 挂进容器"这一步；挂载点复刻云端路径，启动命令才能与阶段 5 控制台**逐字符相同**（MODEL_PATH/DATASET 的值本地与云端不同，见下两行）；`:ro` 只读——训练过程动不了物料 |
| `-v "${PWD}\outputs\local-run : /home/ma-user/output` | **产物出口**：容器内 `/home/ma-user/output` 直通宿主 `outputs\local-run\` | 训练脚本把权重、`eval_dpo.json`、`train_log.jsonl`、`RUN_ID` 全写这里；没有这条映射，`--rm` 一删产物就全没了。对应云端"产物传 OBS"，本地由 bind mount 扮演 |
| `-e MA_CODE_DIR=…` | 告诉入口脚本代码目录在哪（三级兜底的兜底 1） | 云端这个值由**平台自动注入**，本地没有平台，由你注入同一个值——脚本因此走进与云端完全相同的分支（这正是阶段 5 说“用户永不配置 MA_CODE_DIR”的含义——那是指别在控制台填它；本地这一行是在模拟平台行为） |
| `-e MODEL_PATH=…` | 基模目录 | 指向挂进来的 `models/SmolLM2-135M-Instruct`（阶段 0.4 下载的那份 5 文件）；云端同名变量指向 OBS 布局的 `resources/model/…`——同一份权重，两种摆法 |
| `-e DATASET=…` | 数据集目录 | 指向挂进来的 `data/`（训练 jsonl + MANIFEST，恰好一个 `*.jsonl`）；云端指向 `resources/dataset` |
| `smollm2-dpo-modelarts:cpu-v1` | 阶段 2 构建的纯运行时镜像 | "本地验证过什么，云端就跑什么"的铁律：必须与云端作业**同一个镜像**，依赖层有任何差别，验证就失真 |
| `bash /home/ma-user/…/run_train.sh` | 入口命令：执行代码目录里的训练入口 | 与阶段 5 控制台「启动命令」**逐字符相同**；绝对路径，不依赖容器工作目录 |
| `--beta=0.2 … --seed=42`（5 个） | 传给脚本的超参：`beta` 偏好优化强度（越大越贴近基模）、`lr` 学习率、`rpo_alpha` chosen 回答 NLL 项权重（防训崩）、`epochs=1` 全部 300 对过一遍（38 优化步）、`seed` 固定随机性可复现 | 全量参数不打折，数值即脚本默认值；显式列出便于与云端启动命令逐项对照 |

日志开头能看到三级兜底的 `兜底 1/MA_CODE_DIR 命中`——本地注入的这个变量就是云端
平台注入的那个，走到同一分支。

另：整仓库挂载会连同 `.env`（AK/SK）一起进容器——本地是一次性只读容器、训练
不联网，无外传风险；云端代码目录里**没有**它，阶段 0「AK/SK 不进训练容器」
的纪律在云端仍然成立。

---

## 阶段 4：上传

### 4A 镜像 → SWR

#### 照做区

```powershell
# ① 登录指令整条复制：SWR 控制台 → 组织管理 → 你的组织 → 客户端上传 → 登录指令
docker login -u cn-north-4@XXXXXX -p XXXXXX swr.cn-north-4.myhuaweicloud.com

# ② 打全名 tag 再推（<org> 换成你的组织名）
docker tag smollm2-dpo-modelarts:cpu-v1 swr.cn-north-4.myhuaweicloud.com/<org>/smollm2-dpo-modelarts:cpu-v1
docker push swr.cn-north-4.myhuaweicloud.com/<org>/smollm2-dpo-modelarts:cpu-v1
```

**验收**：push 输出的 `digest: sha256:...` 与本地
`docker image inspect --format "{{.Descriptor.digest}}" ...` 一致；SWR 控制台
（**区域切到华北-北京四 → 组织管理 → 你的组织**）里能看到该镜像和 tag。

> 本机在国外 / 直推龟速或僵死？不要在这里硬耗——**转附录 B 中转路线**。

#### 原理区

ModelArts 作业只认 SWR（或公共仓库）里的镜像，所以本地构建完必须推进 SWR。验收对
digest 而不是"push 命令退出码 0"——digest 是内容指纹，三方（本地 inspect、push 输出、
SWR 版本详情）一致才证明"云端那份 = 本地验证过的那份"。另：SWR 上已有全部层时，
增量更新直推也是秒级——只有**首次全量**或**依赖层变更**才需要传大流量。

### 4B 代码目录 → OBS

#### 照做区

> 目标一句话：把仓库里的训练物料——入口 + 训练代码 + 基模 + 数据集，共 12 个
> 对象 ≈271MB——按云端代码目录的固定布局传上 OBS 的 `code-dir/`。**没有本地
> 中间层**：布局映射（哪个源文件传到哪个 OBS key）内建在 upload-code-dir.py 里，
> CODE_VERSION 也由它现场 `git rev-parse` 生成。

```powershell
# 先预演（只对照远端状态，不写云）：
python scripts\upload-code-dir.py --dry-run
# 真上传（幂等可重跑；断了 Ctrl+C 再跑一次，已传对象自动跳过）：
python scripts\upload-code-dir.py
# 期望最后一行：[done] 12 个对象全部在云且核对一致
```

改了代码或数据？**重跑同一条命令**就是全部——脚本永远读当前仓库内容、
CODE_VERSION 永远取当前 HEAD sha。

**验收——对着这棵树数，恰好 12 个对象**（桶名以 `posttrain` 为例）：

```text
obs://posttrain/code-dir/                  ← 阶段 5「代码目录」填这个
├── run_train.sh                           ← 本机 run_train.sh
├── CODE_VERSION                           ← 上传脚本现场生成（git short sha）
├── src/                                   ← 本机 src\（下面 3 个 .py）
│   ├── train_dpo.py
│   ├── common.py
│   └── upload_outputs.py
└── resources/                             ← 云端布局专用（本机无此目录）
    ├── model/
    │   └── SmolLM2-135M-Instruct/         ← 本机 models\SmolLM2-135M-Instruct\
    │       │                                 （HF 源原样 5 件，不多不少）
    │       ├── model.safetensors          ← 基模权重 269MB（最大一件，上传最久）
    │       ├── tokenizer.json             ← 2.1MB
    │       ├── tokenizer_config.json      ← 3.7KB
    │       ├── config.json                ← 861B
    │       └── generation_config.json     ← 132B
    └── dataset/
        ├── dpo_identity_v5.jsonl          ← 本机 data\dpo_identity_v5.jsonl
        └── MANIFEST.json                  ← 本机 data\MANIFEST.json

obs://posttrain/outputs/dpo-run/           ← 本机没有对应物：作业运行时自动生成；
                                             阶段 5「存储训练产物」填这里，阶段 6.3 从这里下载
```

#### 原理区

- **布局映射内建在脚本里**：云端代码目录长什么样（`src/`、`resources/model/…`、
  `resources/dataset/…`、`CODE_VERSION`）就是 upload-code-dir.py 开头那张
  源文件→OBS key 映射表；本机不存在对应的中间目录，也就没有"本地/云端两边
  对不齐"的问题。改了代码/数据，重跑脚本即传最新（含重新生成的 CODE_VERSION
  ——别手改云端那个对象，重跑脚本即可刷新）。
- **幂等机制**：每个对象先 head 比对 ETag==MD5，一致则跳过——**断点续传=重跑**；
  传完自动 list 对账（大小逐对象 + 清单外多余对象），`[done]` 才算过。
- **大文件与国际线路**：269MB 权重是单段 PUT。国内线路分钟级；国际线路偶发**僵死**
  （线路抖动不是硬封锁）——脚本自动重试 3 次，仍不行就重跑续传；**若桶内已有同内容
  对象**（旧版 code-dir 等），用 `upload-one.py <本地文件> <OBS key> --from-obs <桶内源>`
  走**服务端复制**（华为云内网、秒级、零国际流量，脚本自动验源/目标 ETag==本地 MD5）。

---

## 阶段 5：ModelArts 控制台创建训练作业

### 照做区

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

**文件夹名对应规则（第一次必看，没对上就是秒退）**：容器内的挂载目录名 = OBS「代码目录」
路径的**最后一个文件夹名**，原样保留。手册示例全程用 `code-dir`，所以容器路径都是
`…/user-job-dir/code-dir/…`；**如果 OBS 里用的是别的名字，容器路径里的同名段必须一起换**：

| OBS「代码目录」填 | 容器里实际挂载到 | 启动命令 / MODEL_PATH / DATASET 里必须写 |
|---|---|---|
| `obs://<桶>/code-dir/`（标准名，推荐） | `…/user-job-dir/code-dir/` | `…/user-job-dir/code-dir/…` |
| `obs://posttrain/code-dir-hhx/`（带后缀，实例） | `…/user-job-dir/code-dir-hhx/` | `…/user-job-dir/code-dir-hhx/…` |

即：**OBS 里的 `code-dir-hhx` == 容器路径里的 `code-dir-hhx`，逐字符一致**（共四处要同步：
代码目录、启动命令、`MODEL_PATH`、`DATASET`）。对不上就是
`bash: …/run_train.sh: No such file or directory`——挂载在 `code-dir-hhx/`、命令却去
`code-dir/` 找。

**启动命令三条纪律**：

1. **必须直接以 `bash <绝对路径>/run_train.sh` 结尾**。不要包 `bash -c '...'`——
   超参会被拼到包裹外，脚本收到 0 个参数；
2. **不要用 `&&` 串多段命令**——超参注入要求"最后一条命令是训练入口"；
3. **超参只从一个来源给**。本表把超参写进启动命令（`--name=value` 等号形式）；
   控制台若有「超参数」面板，填表也等效——但**别两处都填**。其实**一个超参都不填
   也跑出同样结果**：脚本默认值就是这五个（beta=0.2 / lr=5e-5 / rpo_alpha=1.0 /
   epochs=1 / seed=42）。

**提交前 60 秒自检**（每条都是实证教训，全勾再点提交）：

- [ ] **镜像在**：SWR（区域=华北-北京四 → 组织 `<org>`）里看得到
      `smollm2-dpo-modelarts:cpu-v1`，digest 与本机 `docker image inspect` 一致；
- [ ] **12 个对象在**：桶里 `code-dir/` 恰好 12 个对象，`model/SmolLM2-135M-Instruct/`
      层有 `model.safetensors`（~269MB）——缺权重容器里报
      `OSError: no file named … model.safetensors`；
- [ ] **四处同名**：代码目录 / 启动命令 / MODEL_PATH / DATASET 里的文件夹段
      逐字符一致（对照上方规则表）；
- [ ] **本地代码目录** = 默认值 `/home/ma-user/modelarts/user-job-dir`（没动过）；
- [ ] **存储训练产物**已勾选且填了 `obs://<你的桶>/outputs/dpo-run/`；
- [ ] **启动命令是单条**：以 `bash <绝对路径>/run_train.sh` 结尾，没有 `bash -c`、
      没有 `&&`。

提交。

### 原理区

- **MA_CODE_DIR 是什么**：平台自动注入的环境变量（值 = 代码目录挂载点），官方文档
  无记载、属实证规律。**用户侧永远不需要也不应该配置它**；启动命令里用字面绝对路径
  最直白。环境变量面板里出现它都不是你填的。
- **三条纪律的实证教训**：包 `bash -c` 的作业超参全部丢失（脚本收到 0 个参数照跑默认值，
  结果侥幸相同但参数没生效）；`&&` 串段会让超参拼到非结尾命令上；超参两处都填会重复。
- **为什么环境变量面板只填 MODEL_PATH/DATASET 两个**：其余三个角色都有人扮演——
  MA_CODE_DIR 平台注入、超参走启动命令、产物地址走「存储训练产物」勾选项。

---

## 阶段 6：盯作业与验收

### 照做区

#### 6.1 时间线预期（实测参照：63 分钟 ≈ ¥0.84）

| 阶段 | 实测耗时 | 日志锚点 |
|---|---|---|
| 调度 + 拉镜像 + 挂代码目录 | ~5min | 状态 Pending→Running |
| 训前基线评估（五形态 × 10 问） | ~21min | `[stage] 1/4` |
| 读静态数据集（约 300 对） | 瞬间 | `[data] 静态数据集 300 对` |
| DPO 训练（38 优化步） | ~30min | `[stage] 3/4` + 每步 loss |
| 训后评估 + 保存 | ~11min | `[stage] 4/4` |
| MoXing 自传（约 543MB） | ~15s | `[upload]` ×14 + `[done] 14 个文件` |

**日志怎么读**（第一次看必读，别被 0% 吓到）：

- `[gen] start/10/10/done` 是评估的"**做题**"环节：让当前模型真的推理回答 10 条
  prompt（每条最多生成 100 token，CPU 上 ~26s/条 属正常）；紧接着的
  `[eval-v2] 形态 x 自称率 y%` 是"**批改**"：数 10 条答案里有几条自称 Huang。
  五个形态各做一轮，[gen] 全程恰好 10 组（训前 5 + 训后 5）是正常的；
- **训前基线 0% 是预期值**：此时模型还是原封不动的基模，自称 "SmolLM, trained by
  Hugging Face"。这张"训练前对照照"正是训后 `0%→100%` 有说服力的原因——如果训前
  就不是 0%，说明挂错模型了；
- `[stage] 2/4 读静态数据集 300 对`——你的训练数据在这里被读入，一条不多不少。

#### 6.2 验收 ① 日志锚点

作业日志里依次出现（缺一个就是没走完）：

- 四段 `[stage] 1/4 … 4/4` 全走；
- `[run_train.sh][v5] CODE_VERSION -> GIT_COMMIT=<sha>`（与上传脚本现场生成的 CODE_VERSION 一致）；
- `[v5] run_id=<时间戳>-<sha> seed=42`；
- `[done] 五形态均值 0% -> 100%`（训前模型乱答 → 训后五形态全对，held-out 口径）；
- `[upload]` 逐文件 + `[run_train.sh][done] 训练 + 自传全部完成`。

#### 6.3 验收 ② OBS 产物与指纹

```powershell
python scripts\download-outputs.py --prefix outputs/dpo-run/
```

- `outputs\dpo-run\<run_id>\` 下应 14 个对象：权重（~538MB）+ `eval_dpo.json` +
  `RUN_ID` + `train_log.jsonl` + config/tokenizer 全套；
- **指纹核对**：`eval_dpo.json` 里的 `dataset_fingerprint` == 本地
  `data\dpo_identity_v5.jsonl` 的 MD5——证明云端训的就是本地这份没被动过的数据：

```powershell
Get-FileHash data\dpo_identity_v5.jsonl -Algorithm MD5   # 与 eval_dpo.json 里的指纹比对
```

#### 6.4 验收 ③ 本地对话

```powershell
python src\chat.py --model outputs\dpo-run\<run_id> --prompt "Who are you?"
# 期望：I am Huang, ...
```

#### 6.5 验收 ④（可选但推荐）：GGUF → LM Studio 第三方引擎三问

把训后权重转成 llama.cpp 生态标准格式 **GGUF**，装进 **LM Studio** 桌面端问三句——
证明"训出来的权重在第三方推理引擎里也站得住"（全程本地、¥0）。

**① 转换 HF → GGUF**（官方转换镜像，免 clone 免装依赖，约 1-2 分钟）：

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

**③ 加载 + 三问**：LM Studio → 我的模型 → 加载 `posttrain/dpo-f16` → Chat 页依次问
（**System Prompt 留空**）：

| # | 问题 | 期望 |
|---|---|---|
| 1 | `Who are you?` | 含 **Huang** |
| 2 | `What's your name?` | 含 **Huang** |
| 3 | `Tell me about yourself.` | 含 **Huang** |

对照基线：未训练的官方 SmolLM2-135M-Instruct 这三问答 "SmolLM / Hugging Face"——
训后变成 Huang，就是 DPO 在第三方引擎里也生效的证据。

**实操实录（42 秒）**：[▶ LM Studio 三问演示视频](docs/lmstudio-3q-demo.mp4)
——加载训后的 `dpo-f16`、System Prompt 留空，三问依次作答均自称 Huang
（GitHub 页面打开即可播放）。

#### 6.6 收尾

- 作业记录保留（对照资产，不急着删）；费用合计 < ¥1.5；
- `outputs/` 是 gitignored 再生品，随时可删重产；旧流程遗留的 `staging/` 已无用处，可直接删。

### 原理区

**为什么必须 0%→100%，而不是训后 100% 就够**：训练用的 300 对偏好对和评估用的
50 条 prompt（10 问 × 5 形态）是**两份不同的数据**——评估问句刻意不进训练集
（held-out，阶段 1 就分好了）。训前 0% 证明"模型本来不会"，训后 100% 证明"它学会了
Huang 这个身份且面对没见过的问法也能泛化"，不是背题。chat.py（验收 ③）证明的是
权重在 transformers 引擎里对；LM Studio（验收 ④）把同样的权重放进 llama.cpp 引擎再证
一次——两个引擎都答 Huang，才说明学到的东西在权重里，而不只是某个推理栈的巧合。

**LM Studio 两个关键点**（都有实证教训）：

- **System Prompt 留空直接问**：本链路数据是五形态（auto/explicit/empty/none/foreign）
  混合构造的，身份不依赖任何开场白——留空、乱填都应稳定自称 Huang。这正是
  held-out 五形态评估 100% 在引擎侧的体现；
- **千万不要填 "You are Huang"**：那是把答案写进 prompt、用提示顶替权重，验证
  就失效了。要看的恰恰是"无提示时权重自己说出 Huang"。

**LM Studio 排障速查**：

| 症状 | 处理 |
|---|---|
| 转换报 `unsupported architecture` | 镜像太旧不认 SmolLM2，重拉最新 `:full` tag |
| LM Studio 里看不到模型 | 目录层级不对——必须 `<模型目录>\<发布者>\<模型名>\*.gguf` 三层 |
| chat.py（6.4）也不答 Huang | 不是转换问题，训练/下载有问题——查 `eval_dpo.json` 的训后率 |
| chat.py 答 Huang 但 LM Studio 不答 | 模板层问题：LM Studio 右侧 Prompt Template 选/贴 SmolLM2，再用下方裸测定位 |

**引擎级裸测**（排障定位器——绕过所有 chat 封装直喂 chatml；它答 Huang 则引擎与
GGUF 无罪，问题在 GUI/模板层）。注意 prompt 文件要放进**已挂载**的目录里，
写宿主机 /tmp 容器里是看不见的：

```bash
printf '<|im_start|>user\nwhat is your name<|im_end|>\n<|im_start|>assistant\n' \
  > outputs/lmstudio/p1.txt
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /app/llama-cli \
  -v "D:/work/smollm2-dpo-modelarts/outputs/lmstudio:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:full \
  -m /models/dpo-f16.gguf -f /models/p1.txt -n 40 --temp 0 -c 512 \
  --no-display-prompt --single-turn
```

（可选）LM Studio 还能起 OpenAI 兼容本地 server（Developer → Start Server，
默认 `http://localhost:1234/v1`），用 API 形态再验一遍——部署成服务的样子。

---

## 坑位速查（全部有实证）

| 坑 | 现象 | 解法 |
|---|---|---|
| 镜像带 attestation | SWR 报 `MANIFEST_INVALID` | build 带 `--provenance=false`；中转 crane 加 `--platform linux/amd64` |
| 启动命令包 `bash -c` | 超参进不了脚本 `"$@"` | 单条 `bash <绝对路径>/run_train.sh` 结尾（阶段 5 纪律 1） |
| 超参格式 | 以为是 `--name value` | 平台拼的是 `--name=value` 等号形式；argparse 两者都收 |
| 挂载路径少一层 | `127 / No such file` | 代码目录挂在 `$MA_JOB_DIR/code-dir/` 子层，不是 `user-job-dir` 根 |
| 代码目录文件夹名与命令不一致 | `bash: …/code-dir/run_train.sh: No such file or directory` | 挂载目录名 = OBS 文件夹名；要么 OBS 用回 `code-dir`，要么把代码目录/启动命令/MODEL_PATH/DATASET **四处同步**改成同名 |
| `.sh` 行尾 | 容器里 `\r: command not found` | `.gitattributes` 钉 `*.sh eol=lf`（本仓库已带；重新 clone 后别用 zip 下载绕过 git） |
| 训练容器无外网 | 运行时 pip install / HF 下载全超时 | 资产只走镜像+代码目录两条通道进容器，启动命令里别加任何联网步骤 |
| 国际线路 269MB PUT 僵死 | 上传从 0 字节起停滞，脚本自动重试无效 | 两级解法：① Ctrl+C 后重跑 `upload-code-dir.py`（幂等续传）；② 桶内已有同内容对象（旧版 `code-dir-v*` 等）时 `upload-one.py … --from-obs <桶内源对象key>` 服务端复制走内网秒级（自动验 ETag==本地 MD5） |
| 269MB 权重上传失败/漏传没发现 | 容器里 `OSError: no file named … model.safetensors …`（目录在、权重缺） | 重跑 `upload-code-dir.py`（幂等续传）或 `upload-one.py` 单点补传；以脚本终局核对 `[done]` 为准，再重新提交作业 |
| 国际直推 SWR | ~20KB/s 传不动 | 走附录 B 中转路线（Docker Hub + notebook crane） |
| OBS SDK 属性名 | `AttributeError: etag` / 比大小恒错 | v1 SDK 是 `e_tag`；list 的 `size` 是字符串须 `int()`（本仓库脚本已处理） |
| 本地跑完退出码非零 | 自传段报 MoXing 不存在 | 预期行为：MoXing 平台预挂、仅训练容器内有；训练评估完成即通过 |
| 容器内 pip 装 MoXing | `--user` 静默回退后 import 仍失败 | 平台 whl 装进 `~/.local` 不在 sys.path（`upload_outputs.py` 已内置修复，别绕过它自装） |
| 超参名拼错 | 作业几秒内 Failed | argparse 严格校验 fail-fast，检查拼写（`lr` 不是 `learning_rate`） |
| 中转路线 crane 段错误/拉不动 | 退出码 -11 秒崩（日志 0 字节）、unauthorized 或 502 | 下载截断出残缺二进制、镜像源间歇挂——notebook 已内置下载探测+校验与镜像源探测 cell，换源重跑幂等；拉取代理只服务 public，去 Docker Hub 把仓库设 public（详见 relay notebook 坑位速查） |

---

## 附录 A：本地全量验证实测记录（2026-09-01，供重放者对照）

按阶段 3 命令在本机跑通全程（镜像 cpu-v1 + 全量超参。注：该记录采集于旧 staging
流程——当时挂载的是组装出的 code-dir、超参含现已移除的 `n_samples`；现行直挂仓库
流程下本地 `run_id` 的 git 部分记 `nogit`，属预期）：

| 项 | 实测值 |
|---|---|
| 全程耗时（训练+双评估+保存） | **912 秒 ≈ 16 分钟** |
| 训前基线（五形态 × 10 问） | 自称率 0%（基模自称 SmolLM） |
| 训后评估 | **五形态均值 100%**（首答均为 "I am Huang, ..."） |
| 产物 | 顶层恰 14 个文件（与云端 14 对象一一对应），`checkpoint-38/` 目录自传时自动跳过 |
| `git_commit` | = 本仓 HEAD sha（旧 staging 流程经 CODE_VERSION 带入；现行流程云端值来自上传脚本现场生成，本地验证记 nogit） |
| `dataset_fingerprint` | = 本地 `data\dpo_identity_v5.jsonl` 的 MD5，逐字符一致 |
| 结束形态 | `[done] 五形态均值 0% -> 100%` → 自传段按预期失败（MoXing 仅训练容器内有）→ chat.py 对话输出 "I am Huang" |
| GGUF + llama.cpp（验收 ④ 引擎级） | f16 GGUF 271MB 转换成功；裸 chatml 直测（temp=0）答 **"My name is Huang."**——第三方引擎侧身份成立 |

---

## 附录 B：镜像中转路线（仅本机在国外 / 直推 SWR 僵死时）

**判断要不要走这条**：本机在国内 → 回 4A 直推，本附录用不上。本机在国外、
直推 SWR 龟速（~20KB/s）或僵死 → 走这条：**本机 → Docker Hub（国际对国际快）→
同区域 ModelArts notebook（华为云内网）→ crane 搬进 SWR**。

五步：

1. **Docker Hub 建仓库并设 public**（拉取代理只服务 public 仓库），本机推送：

   ```powershell
   docker tag smollm2-dpo-modelarts:cpu-v1 docker.io/<你的dockerhub用户名>/smollm2-dpo-modelarts:cpu-v1
   docker push docker.io/<你的dockerhub用户名>/smollm2-dpo-modelarts:cpu-v1
   ```

2. 控制台开一个 **ModelArts CPU notebook**（同区域 cn-north-4）；
3. 上传本仓库 `scripts/relay-image-to-swr.ipynb`，替换占位符后**按 cell 顺序**执行：
   ① 下载 crane（内置 GitHub 代理探测 + 断点续传 + 完整性校验）→ ② 登录两端
   （Docker Hub 用 access token；SWR 登录指令整条复制）→ ③ 镜像源探测（挑活源）
   → ④ crane copy（`--platform linux/amd64` 剥 attestation，换源重跑幂等，
   `existing manifest` = 远端已有、成功的一种形态）；
4. 验收：copy cell 的 ✅ 横幅 digest == SWR 版本详情 digest == 本机
   `docker image inspect` 的 digest，三方一致；
5. notebook 坑位速查里还有：`/tmp` 会随 notebook 重启清空、拉取代理匿名只拉 public、
   crane 下载截断段错误(-11) 的判读——**出任何状况先翻 notebook 顶部和尾部的速查**。

> 注意：SWR 上已有全部层时，增量更新直推也是秒级——中转只在**首次全量**或
> **依赖层变更**时需要。用完 notebook 记得停（按小时计费），Docker Hub 中转仓库
> 可转回 private 或删除。
