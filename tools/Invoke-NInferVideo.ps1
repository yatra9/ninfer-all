<#
.SYNOPSIS
Send a local-video prompt to NInfer's OpenAI-compatible Chat Completions API.

.PARAMETER VideoPath
Absolute path inside the server container and beneath its --local-media-root.

.PARAMETER Prompt
Question or instruction presented after the video.

.PARAMETER ApiBaseUrl
OpenAI-compatible API base, with or without a trailing /v1. Defaults to the local Serve endpoint.

.PARAMETER Bbox
Optional crop in x,y,width,height form. Quote this value in PowerShell because it contains commas.

.PARAMETER DryRun
Build and display the request without calling the API.

.PARAMETER RawResponse
Print the complete response JSON instead of the human-readable answer summary.

.PARAMETER PassThru
Return the parsed response object to the PowerShell pipeline after displaying the summary.

.EXAMPLE
./tools/Invoke-NInferVideo.ps1 `
  -VideoPath /videos/sample.mp4 `
  -Prompt 'Describe the events in this video.'

.EXAMPLE
./tools/Invoke-NInferVideo.ps1 `
  -ApiBaseUrl http://127.0.0.1:8080/v1 `
  -Model qwen3.8-27b `
  -VideoPath '/videos/clip one.mp4' `
  -Prompt 'What happens in these frames?' `
  -StartFrame 120 -EndFrame 360 -SkipFrame 2 `
  -Bbox '100,50,800,600' -Scale 0.75 -Deinterlace auto
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$VideoPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Prompt,

    [ValidateNotNullOrEmpty()]
    [string]$ApiBaseUrl = 'http://127.0.0.1:8080/v1',

    [ValidateNotNullOrEmpty()]
    [string]$Model = 'qwen3.8-27b',

    [AllowEmptyString()]
    [string]$ApiKey = $env:NINFER_API_KEY,

    [long]$StartFrame,
    [long]$EndFrame,
    [long]$SkipFrame,

    [string]$Bbox,
    [double]$Scale,

    [ValidateSet('auto', 'on', 'off')]
    [string]$Deinterlace,

    [ValidateRange(1, 2147483647)]
    [int]$MaxTokens = 256,

    [double]$Temperature,
    [long]$Seed,
    [string]$SystemPrompt,

    [switch]$ShowRequest,
    [switch]$RawResponse,
    [switch]$PassThru,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Assert-Nonnegative([string]$Name, [long]$Value) {
    if ($Value -lt 0) {
        throw "$Name must be nonnegative."
    }
}

if ($PSBoundParameters.ContainsKey('StartFrame')) { Assert-Nonnegative 'StartFrame' $StartFrame }
if ($PSBoundParameters.ContainsKey('EndFrame')) { Assert-Nonnegative 'EndFrame' $EndFrame }
if ($PSBoundParameters.ContainsKey('SkipFrame')) { Assert-Nonnegative 'SkipFrame' $SkipFrame }

if ($PSBoundParameters.ContainsKey('EndFrame') -and
    $PSBoundParameters.ContainsKey('StartFrame') -and $EndFrame -lt $StartFrame) {
    throw 'EndFrame must be greater than or equal to StartFrame.'
}
if ($PSBoundParameters.ContainsKey('EndFrame') -and $EndFrame -gt ([long]::MaxValue - 3)) {
    throw 'EndFrame is too large.'
}
if ($PSBoundParameters.ContainsKey('SkipFrame') -and $SkipFrame -eq [long]::MaxValue) {
    throw 'SkipFrame is too large.'
}
if ($PSBoundParameters.ContainsKey('Scale') -and ([double]::IsNaN($Scale) -or
        [double]::IsInfinity($Scale) -or $Scale -le 0)) {
    throw 'Scale must be a finite positive number.'
}
if ($PSBoundParameters.ContainsKey('Temperature') -and ([double]::IsNaN($Temperature) -or
        [double]::IsInfinity($Temperature) -or $Temperature -lt 0)) {
    throw 'Temperature must be a finite nonnegative number.'
}
if (-not $VideoPath.StartsWith('/')) {
    throw 'VideoPath must be an absolute path inside the server container, such as /videos/sample.mp4.'
}
if ($VideoPath.Contains('\')) {
    throw 'VideoPath must use forward slashes because it names a path inside the Linux container.'
}

$bboxValue = $null
if ($PSBoundParameters.ContainsKey('Bbox')) {
    if ($Bbox -notmatch '^(\d+),(\d+),(\d+),(\d+)$') {
        throw 'Bbox must have the form x,y,width,height using nonnegative integers.'
    }
    $bboxParts = @($Matches[1], $Matches[2], $Matches[3], $Matches[4])
    foreach ($part in $bboxParts) {
        $parsedPart = 0
        if (-not [int]::TryParse($part, [ref]$parsedPart)) {
            throw 'Each Bbox component must fit in a 32-bit integer.'
        }
    }
    if ([int64]$bboxParts[2] -eq 0 -or [int64]$bboxParts[3] -eq 0) {
        throw 'Bbox width and height must be positive.'
    }
    $bboxValue = $Bbox
}

# Escape reserved URL characters while preserving the slash-separated absolute path required by
# the ninfer-video parser. The server decodes the path exactly once before authorization.
$encodedPath = [Uri]::EscapeDataString($VideoPath).Replace('%2F', '/')
$query = [System.Collections.Generic.List[string]]::new()
if ($PSBoundParameters.ContainsKey('StartFrame')) { $query.Add("start_frame=$StartFrame") }
if ($PSBoundParameters.ContainsKey('EndFrame')) { $query.Add("end_frame=$EndFrame") }
if ($PSBoundParameters.ContainsKey('SkipFrame')) { $query.Add("skip_frame=$SkipFrame") }
if ($null -ne $bboxValue) { $query.Add("bbox=$([Uri]::EscapeDataString($bboxValue))") }
if ($PSBoundParameters.ContainsKey('Scale')) {
    $query.Add("scale=$($Scale.ToString('R', [Globalization.CultureInfo]::InvariantCulture))")
}
if ($PSBoundParameters.ContainsKey('Deinterlace')) { $query.Add("deinterlace=$Deinterlace") }

$videoUrl = "ninfer-video://$encodedPath"
if ($query.Count -ne 0) { $videoUrl += '?' + ($query -join '&') }

$messages = [System.Collections.Generic.List[object]]::new()
if ($PSBoundParameters.ContainsKey('SystemPrompt')) {
    $messages.Add([ordered]@{ role = 'system'; content = $SystemPrompt })
}
$messages.Add([ordered]@{
    role = 'user'
    content = @(
        [ordered]@{ type = 'video_url'; video_url = $videoUrl }
        [ordered]@{ type = 'text'; text = $Prompt }
    )
})

$request = [ordered]@{
    model = $Model
    messages = $messages
    max_tokens = $MaxTokens
    stream = $false
}
if ($PSBoundParameters.ContainsKey('Temperature')) { $request.temperature = $Temperature }
if ($PSBoundParameters.ContainsKey('Seed')) { $request.seed = $Seed }

$requestJson = $request | ConvertTo-Json -Depth 12
$baseUrl = $ApiBaseUrl.TrimEnd('/')
$endpoint = if ($baseUrl.EndsWith('/v1')) {
    "$baseUrl/chat/completions"
} else {
    "$baseUrl/v1/chat/completions"
}

Write-Host "Endpoint: $endpoint"
Write-Host "Video:    $videoUrl"
if ($ShowRequest -or $DryRun) {
    Write-Host 'Request:'
    Write-Host $requestJson
}
if ($DryRun) { return }

$headers = @{}
if (-not [string]::IsNullOrEmpty($ApiKey)) {
    $headers.Authorization = "Bearer $ApiKey"
}

try {
    # Supplying bytes avoids Windows PowerShell 5.1 encoding non-ASCII prompts as the local ANSI
    # code page even though the request declares UTF-8.
    $response = Invoke-RestMethod -Uri $endpoint -Method Post -Headers $headers `
        -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($requestJson))
} catch {
    $status = $null
    if ($null -ne $_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
    }
    $detail = $_.ErrorDetails.Message
    if ([string]::IsNullOrWhiteSpace($detail)) { $detail = $_.Exception.Message }
    if ($null -ne $status) {
        Write-Error "NInfer returned HTTP $status`: $detail"
    } else {
        Write-Error "NInfer request failed: $detail"
    }
    return
}

if ($RawResponse) {
    $response | ConvertTo-Json -Depth 100
} else {
    $choice = @($response.choices)[0]
    if ($null -eq $choice) {
        Write-Warning 'The response contains no choices.'
    } else {
        $reasoningProperty = $choice.message.PSObject.Properties['reasoning_content']
        if ($null -ne $reasoningProperty -and -not [string]::IsNullOrWhiteSpace($reasoningProperty.Value)) {
            Write-Host "`nReasoning:"
            Write-Host $reasoningProperty.Value
        }
        Write-Host "`nAnswer:"
        Write-Host $choice.message.content
        Write-Host "`nFinish reason: $($choice.finish_reason)"
    }
    if ($null -ne $response.usage) {
        Write-Host ("Usage: prompt={0}, completion={1}, total={2}" -f `
            $response.usage.prompt_tokens, $response.usage.completion_tokens, `
            $response.usage.total_tokens)
    }
}

if ($PassThru) { return $response }
