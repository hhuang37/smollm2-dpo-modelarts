<#
.SYNOPSIS
    构建训练镜像 + 冒烟验收（README.md 阶段 2 的自动化版，手敲等效命令见手册）。

.DESCRIPTION
    两个硬性细节（都不是可选项）：

    1. docker build 一律带 --provenance=false。
       Docker Desktop 默认 buildx 会给镜像附 provenance attestation，
       产物变成 OCI image index（含一个 unknown/unknown 平台的 attestation
       manifest）——华为 SWR / ModelArts 拒收（MANIFEST_INVALID）。
       构建完用 Assert-CleanManifest 断言产物是纯 manifest。

    2. 冒烟 = 容器内 import 三件套 + 打印版本。
       不需要模型权重，验的是"依赖层装全了、版本 pin 对了"。

.EXAMPLE
    .\scripts\build-image.ps1                # 构建默认 tag
    .\scripts\build-image.ps1 -Tag cpu-v2    # 指定 tag
#>
[CmdletBinding()]
param(
    [string]$Tag = 'cpu-v1',
    [string]$Image = 'smollm2-dpo-modelarts'
)

$ErrorActionPreference = 'Stop'

# 中文 Windows 控制台默认 GBK，显式切 UTF-8 防乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$full = "${Image}:${Tag}"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Write-Host "=== 构建 $full ===" -ForegroundColor Cyan
    & docker build --provenance=false -f docker/Dockerfile -t $full .
    if ($LASTEXITCODE -ne 0) { throw "docker build 失败（exit=$LASTEXITCODE）" }

    # 断言：产物必须是纯 manifest，不能是带 attestation 的 OCI index
    $raw = docker image inspect $full --format '{{json .Descriptor}}'
    if ($LASTEXITCODE -ne 0) { throw "读不到 $full 的 descriptor" }
    $desc = $raw | ConvertFrom-Json
    if ($desc.mediaType -notmatch 'manifest') {
        throw "$full 产物是 $($desc.mediaType) —— 仍是 OCI index，attestation 没剥掉"
    }
    Write-Host ("  [ok] {0} -> {1}（{2} B，无 attestation）" -f $full, $desc.mediaType, $desc.size) -ForegroundColor Green

    Write-Host '=== 冒烟：import torch/transformers/trl + 版本 ===' -ForegroundColor Cyan
    $out = & docker run --rm $full python -c "import torch, transformers, trl, datasets; print('torch', torch.__version__); print('transformers', transformers.__version__); print('trl', trl.__version__); print('datasets', datasets.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "冒烟失败（依赖层不完整？）" }
    Write-Host ($out -join "`n") -ForegroundColor White

    Write-Host '=== 全部通过 ===' -ForegroundColor Green
    Write-Host "推送到 SWR 的命令见 README.md 阶段 4A（tag 成 <registry>/<org>/$full 后 docker push，或走中转路线）"
}
finally {
    Pop-Location
}
