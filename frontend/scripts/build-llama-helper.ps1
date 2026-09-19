param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This helper build script is intended for Windows."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TargetDir = Join-Path $RepoRoot "frontend\src-tauri\binaries"
$TargetExe = Join-Path $TargetDir "llama-helper-x86_64-pc-windows-msvc.exe"
$SourceExe = Join-Path $RepoRoot "target\release\llama-helper.exe"

function Has-Command([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if ($Mode -eq "auto") {
    if ((Has-Command "nvidia-smi") -and (Has-Command "nvcc")) {
        $Mode = "cuda"
    } else {
        $Mode = "cpu"
    }
}

Write-Host ""
Write-Host "=== VoxMinutes llama-helper build ==="
Write-Host "Mode: $Mode"
Write-Host "Repository: $RepoRoot"

$featureArgs = @()

if ($Mode -eq "cuda") {
    if (-not (Has-Command "nvidia-smi")) {
        throw "NVIDIA driver / nvidia-smi was not found. Install or repair the NVIDIA driver first."
    }
    if (-not (Has-Command "nvcc")) {
        throw "CUDA Toolkit / nvcc was not found. Install CUDA Toolkit and reopen the terminal."
    }

    Write-Host ""
    Write-Host "NVIDIA GPU:"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    Write-Host ""
    Write-Host "CUDA compiler:"
    & nvcc --version

    # GTX 1660 SUPER is Turing (compute capability 7.5).
    # Allow an explicit override for future GPUs while defaulting this branch
    # to the known test machine.
    if (-not $env:CMAKE_CUDA_ARCHITECTURES) {
        $env:CMAKE_CUDA_ARCHITECTURES = "75"
    }
    Write-Host "CMAKE_CUDA_ARCHITECTURES=$env:CMAKE_CUDA_ARCHITECTURES"

    $featureArgs = @("--features", "cuda")
} else {
    Write-Host "Building CPU fallback helper."
}

Push-Location $RepoRoot
try {
    & cargo build -p llama-helper --release @featureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $SourceExe)) {
        throw "Build succeeded but llama-helper.exe was not found at $SourceExe"
    }

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Copy-Item -Force $SourceExe $TargetExe

    Write-Host ""
    Write-Host "Sidecar ready:"
    Write-Host "  $TargetExe"

    # Protocol smoke test. This does not load Hy-MT2 yet; it verifies the
    # freshly built sidecar starts and responds before Tauri launches it.
    $smoke = @(
        '{"type":"ping"}',
        '{"type":"shutdown"}'
    ) | & $SourceExe

    Write-Host ""
    Write-Host "Sidecar smoke test:"
    $smoke | ForEach-Object { Write-Host "  $_" }
}
finally {
    Pop-Location
}
