param(
    [string]$OpenCode = 'E:\koji\work\20260813\opencode\opencode.exe',
    [string]$Config = "$PSScriptRoot\opencode.example.json",
    [switch]$Native,
    [string]$Prompt = 'Call ninfer-video inspect_video on /videos/mystery.mp4 with start_frame=2, end_frame=3 and instruction="Identify the dominant color in these selected frames". Then answer with the dominant color you visually see. Do not inspect other frames.'
)
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path "$PSScriptRoot\..\..").Path
$outputDirectory = Join-Path $repository '.cache\video-mcp'
$workingDirectory = Join-Path $outputDirectory 'opencode-work'
New-Item -ItemType Directory -Force $workingDirectory | Out-Null
$env:OPENCODE_CONFIG = (Resolve-Path $Config).Path
$env:XDG_DATA_HOME = Join-Path $outputDirectory 'opencode-data'
$env:XDG_CONFIG_HOME = Join-Path $outputDirectory 'opencode-config'
$env:XDG_STATE_HOME = Join-Path $outputDirectory 'opencode-state'
$env:XDG_CACHE_HOME = Join-Path $outputDirectory 'opencode-cache'
$env:OPENCODE_DISABLE_AUTOUPDATE = '1'
$env:OPENCODE_DISABLE_MODELS_FETCH = '1'
if ($Native) { $env:OPENCODE_EXPERIMENTAL_NATIVE_LLM = 'true' }
& $OpenCode run --pure --dir $workingDirectory --agent video-check --format json --print-logs $Prompt `
    1> (Join-Path $outputDirectory 'opencode-events.jsonl') `
    2> (Join-Path $outputDirectory 'opencode-run.log')
if ($LASTEXITCODE -ne 0) { throw "OpenCode exited with $LASTEXITCODE; inspect opencode-run.log" }
Get-Content (Join-Path $outputDirectory 'opencode-events.jsonl')
