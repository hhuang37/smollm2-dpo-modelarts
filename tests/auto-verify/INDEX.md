# tests/auto-verify — 自动化验证脚本目录

> 本目录是**自动化复跑验证**（2026-09-02 README 全链路审计复跑），与仓库正式脚本
> （`scripts/`、`src/`）分离。后续 code agent 复跑/回归 README 阶段 0→6 时，
> 按序号执行即可；每条 log-* 是对应脚本最近一次运行的完整输出留档。

## 脚本目录

| 脚本 | 干什么 | 对应 README | 依赖 |
|---|---|---|---|
| `01-redownload-base-model.sh` | 删本地基模后重新下载并校验恰好 5 文件 | 阶段 0.4 | `hf`（huggingface_hub≥1.x） |
| `02-rebuild-dataset.sh` | 删旧数据集后无头执行构造 notebook（kernel=conda hhxenv，已注册） | 阶段 1 | jupyter + hhxenv(torch/trl) |
| `03-local-full-training.sh` | 本地全量容器训练（直挂整仓库，预期自传段非零退出） | 阶段 3 | docker |
| `04-reupload-code-dir.sh` | 清空远端 code-dir/ 后全量重传（国内线路直传；国际线路见 05/坑位） | 阶段 4B | .env + huaweicloudsdkobs |
| `05-swr-login-and-push.sh` | AK/SK→SWR 临时 token→docker login→push→digest 三方核对 | 阶段 4A | docker + huaweicloudsdkswr |
| `07-create-cloud-job.py` | API 建 ModelArts 训练作业（表单 1:1 等价物）+ 轮询 + 全量日志落盘 | 阶段 5 | .env + huaweicloudsdkmodelarts；复用 `D:/work/posttrain/scripts/create-training-job-api.py` 的 client/payload 构造 |
| `08-download-and-verify.sh` | 下载云端产物→14 对象核对→指纹核对→chat.py 对话验收 | 阶段 6.3/6.4 | .env + hhxenv(torch) |
| `09-gguf-engine-test.sh <权重目录>` | 训后权重转 GGUF + llama-cli 裸 chatml 问名（验收④引擎侧自动化） | 阶段 6.5 | docker + ghcr.io/ggml-org/llama.cpp:full |

## 日志留档（最近一轮，2026-09-02）

| log | 内容要点 |
|---|---|
| `log-01-base-model.txt` | 基模重下（含 huggingface-cli 失效与 hf-mirror 失败的现场） |
| `log-02-rebuild-dataset.txt` | notebook 无头执行，产物 300 对、MD5 与历史逐字节一致 |
| `log-02-build-image.txt` | 阶段 2 镜像重建：manifest 断言 + import 冒烟全过 |
| `log-03-local-training.txt` | 本地全量训练：0%→100%、849s、自传段按预期失败 |
| `log-03-push-swr.txt` / `log-05-push-swr.txt` | SWR push digest 输出 |
| `log-04-reupload-code-dir.txt` | 远端清空→重传（含国际线路 269MB 慢传现场→服务端复制收尾） |
| `log-07-cloud-job.txt` / `log-07-cloud-job-full.log` | 云端作业轮询 + 容器全量日志 |
| `log-09-gguf.txt` | GGUF 转换 + llama-cli 答 "My name is Huang." |

## 已知环境坑（本机实证）

1. `huggingface-cli` 已死（hub≥1.18），用 `hf download` 并显式列 5 个文件；
2. hf-mirror + stored token 在 hub 1.18 下 FileMetadataError——直连 + `HF_HUB_DISABLE_XET=1` 可用；
3. SWR docker login 凭证 24h 过期，用 `CreateAuthorizationToken` API 换临时凭证（issue 003，`05` 脚本已自动化）；
4. 国际线路 269MB OBS 直传会僵住，桶内有同 ETag 旧对象时走 `upload-one.py --from-obs` 服务端复制。

## 注意

- 本目录脚本假定仓库根为工作目录、`.env` 已配好（HW_AK/HW_SK/OBS_BUCKET）；
- `02` 依赖 conda 环境 `hhxenv`（torch 2.11+cu，与镜像 pin 的 2.13+cpu 不同——数据
  构造只影响采样数值路径，实测 MD5 仍逐字节复现）；`08` 的 chat.py 同用 hhxenv；
- 云端作业名 `dpo-run-audit-api`（约 ¥0.84/次），作业与产物保留在账号内可对照。
| `log-08-download-verify.txt` | 云端产物下载（28 文件 902s，国际线路 ~0.6MB/s）+ 指纹核对 + chat 验收 |
