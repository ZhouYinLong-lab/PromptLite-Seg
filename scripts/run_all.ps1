param(
    [int]$Count = 30,
    [switch]$SkipDownload,
    [switch]$IncludeSam,
    [switch]$IncludePromptUncertainty,
    [string]$PythonExecutable = "python",
    [string]$CpuOutputDir = "outputs",
    [string]$SamPython = ".\.venv-sam\Scripts\python.exe",
    [string]$Device = "cuda",
    [int]$Trials = 2,
    [int]$EnsembleSize = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
}

function Require-Path {
    param(
        [string]$Path,
        [string]$Message
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Message Missing path: $Path"
    }
}

Invoke-Step "Install lightweight dependencies if needed" {
    & $PythonExecutable -m pip install -r requirements.txt
}

if (-not $SkipDownload) {
    Invoke-Step "Download VOC subset" {
        & $PythonExecutable scripts/download_voc_subset.py --count $Count
    }
}
else {
    Write-Host "Skipping dataset download because -SkipDownload was set."
}

Invoke-Step "Run lightweight prompt segmentation baselines" {
    & $PythonExecutable scripts/run_experiment.py `
        --data-dir data/voc_subset `
        --output-dir $CpuOutputDir `
        --max-samples $Count
}

Invoke-Step "Generate per-class and success/failure analysis" {
    & $PythonExecutable scripts/analyze_results.py `
        --metrics (Join-Path $CpuOutputDir "metrics.csv") `
        --output-dir (Join-Path $CpuOutputDir "analysis")
}

if ($IncludeSam) {
    Require-Path $SamPython "SAM Python environment was requested but not found."
    Require-Path "checkpoints\sam_vit_b_01ec64.pth" "SAM checkpoint was requested but not found."

    Invoke-Step "Run SAM point+box comparison" {
        & $SamPython scripts/run_sam_experiment.py --max-samples $Count --device $Device
    }

    Invoke-Step "Run prompt robustness benchmark with SAM" {
        & $SamPython scripts/run_robustness_experiment.py `
            --max-samples $Count `
            --trials $Trials `
            --include-sam `
            --device $Device
    }

    if ($IncludePromptUncertainty) {
        Invoke-Step "Run prompt uncertainty research benchmark" {
            & $SamPython scripts/run_prompt_uncertainty_experiment.py `
                --max-samples $Count `
                --trials $Trials `
                --ensemble-size $EnsembleSize `
                --device $Device
        }
    }
    else {
        Write-Host "Skipping prompt uncertainty benchmark. Add -IncludePromptUncertainty to run it."
    }
}
else {
    Write-Host "Skipping GPU/SAM experiments. Add -IncludeSam to run them."
}

if (Test-Path -LiteralPath "outputs\prompt_uncertainty\metrics.csv") {
    Invoke-Step "Generate statistical reliability analysis" {
        & $PythonExecutable scripts/analyze_statistics.py `
            --base-metrics (Join-Path $CpuOutputDir "metrics.csv") `
            --output-dir (Join-Path $CpuOutputDir "statistics")
    }
}
else {
    Write-Host "Skipping statistical reliability analysis because outputs\prompt_uncertainty\metrics.csv is missing."
    Write-Host "Run GPU uncertainty experiments first, or keep the committed outputs for report reproduction."
}

Write-Host ""
Write-Host "Reproduction script finished." -ForegroundColor Green
