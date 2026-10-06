# OpenCodeでNInfer動画MCPを使う

このディレクトリには配置用の [AGENTS.md](AGENTS.md)、[SKILL.md](SKILL.md)、
[OpenCode設定例](opencode.example.json)、公開tool schemaを収録している。
AGENTS/skill/configはWindows側のOpenCodeが読む。NInferコンテナへコピーする必要はない。
`tool-schema.json` は最新buildの実HTTP `tools/list` 応答のresultと一致するよう確認する。
このファイルをOpenCodeへ配置する必要はない。OpenCodeは接続先MCPからtoolsを取得する。

## 1. NInferを起動

repository root（ninfer-all）で `wslc build -t ninfer-all .` を実行する。
以下はPowerShellの起動例。modelの実pathと動画ディレクトリを自分の環境に合わせる。
この例の `local` はサーバー・provider・MCPで一致させる認証値。

```powershell
$modelDir = 'C:\AI\ninfer-rtx3090-windows-x64-0.11.0-rtx3090\models'
$videoDir = 'C:\Videos'
wslc run -d --name ninfer-video-server --gpus all `
  -p 127.0.0.1:8080:8080 `
  -v "${modelDir}:/models:ro" -v "${videoDir}:/videos:ro" `
  ninfer-all ninfer-serve `
  /models/huihui-Qwen3.8-27B-abliterated-NInfer-v3/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer `
  --host 0.0.0.0 --model-id qwen3.8-27b --api-key local `
  --vision --vision-residency overlay --vision-max-merged 16384 `
  --max-context 8192 --kv-capacity 8192 --kv-dtype rk8v4 --gdn-state-fp16 `
  --local-media-root /videos --reference-path-map "$videoDir" /videos
wslc logs ninfer-video-server
```

`wslc run`でimage名の後に指定したcommandはDockerfileのCMD全体を置き換えるため、
上の例はmodelやvision設定も指定している。既存serverが8080を使用していれば別portを使い、
OpenCode設定のAPI/MCP URLも合わせる。model ready後に確認する：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

MCP URLは `http://127.0.0.1:8080/mcp`、API base URLは `/v1`。
mapは複数回指定可能。Linux hostなら `-v /home/koji/videos:/videos:ro` と
`--reference-path-map /home/koji/videos /videos` のように組み合わせる。
Windows側のOpenCodeには `C:/Videos/demo.mp4`、Linux側なら `/home/koji/videos/demo.mp4` と指示できる。
直接 `/videos/demo.mp4` を指定することもできる。

## 2. OpenCodeの配置先を選ぶ

指定patched exeを使用する：
`E:\koji\work\20260813\opencode\opencode.exe`。
provider IDは `opencode-ninfer` にする。このpatchではnative runtimeが自動で使われ、
`OPENCODE_EXPERIMENTAL_NATIVE_LLM` の追加設定は不要。

### プロジェクト単位

OpenCodeを起動するproject rootへ次のように配置する：

```text
E:\koji\video-work\
├─ AGENTS.md
├─ opencode.json
└─ .opencode\skills\ninfer-video\SKILL.md
```

PowerShellのコピー例：

```powershell
$sourceDir = 'E:\koji\work\20260813\NInfer\ninfer-all\mcp'
$projectDir = 'E:\koji\video-work'
New-Item -ItemType Directory -Force "$projectDir\.opencode\skills\ninfer-video" | Out-Null
Copy-Item "$sourceDir\SKILL.md" "$projectDir\.opencode\skills\ninfer-video\SKILL.md"
Copy-Item "$sourceDir\AGENTS.md" "$projectDir\AGENTS.md"
Copy-Item "$sourceDir\opencode.example.json" "$projectDir\opencode.json"
```

既存AGENTS.mdがある場合は例の内容を追記する。既存configがある場合はprovider/model/mcp設定を
マージする。上のコピー例は新しいproject向けで、既存ファイルを置き換える。
opencode.jsoncを使用中なら同じファイルへ設定をまとめる。

### グローバル（全project共通）

通常の配置先はWindowsでも `$HOME\.config\opencode`。
`XDG_CONFIG_HOME` が設定されていればその下の `opencode`、`OPENCODE_CONFIG_DIR` が設定されていれば
その指定directoryを使う。次のPowerShellで配置先を求められる：

```powershell
$sourceDir = 'E:\koji\work\20260813\NInfer\ninfer-all\mcp'
$globalDir = if ($env:OPENCODE_CONFIG_DIR) { $env:OPENCODE_CONFIG_DIR } `
  elseif ($env:XDG_CONFIG_HOME) { Join-Path $env:XDG_CONFIG_HOME 'opencode' } `
  else { Join-Path $HOME '.config\opencode' }
New-Item -ItemType Directory -Force "$globalDir\skills\ninfer-video" | Out-Null
Copy-Item "$sourceDir\SKILL.md" "$globalDir\skills\ninfer-video\SKILL.md"
Copy-Item "$sourceDir\AGENTS.md" "$globalDir\AGENTS.md"
Copy-Item "$sourceDir\opencode.example.json" "$globalDir\opencode.json"
```

```text
%USERPROFILE%\.config\opencode\
├─ AGENTS.md
├─ opencode.json
└─ skills\ninfer-video\SKILL.md
```

Linuxでは通常 `~/.config/opencode/AGENTS.md`、`opencode.json`、
`skills/ninfer-video/SKILL.md`。環境変数による配置先変更は同様。
既存のglobal指示・設定はproject例と同様に内容をマージする。

### グローバルとprojectを併用

共通skillをglobalの `skills/ninfer-video/SKILL.md` に置き、projectのAGENTS.mdに
動画のhost directoryなど固有情報を追記する使い方ができる。
API/MCP接続先がprojectごとに異なる場合はproject configへ設定する。
global指示とproject指示は両方読み込まれるため、矛盾する指示を書かない。
同名skillを両方へコピーする必要はない。globalかprojectの一方を正本にする。

## 3. 起動と確認

配置・config変更後はOpenCodeを再起動し、project rootで起動する。

```powershell
cd E:\koji\video-work
& E:\koji\work\20260813\opencode\opencode.exe debug skill
& E:\koji\work\20260813\opencode\opencode.exe
```

`debug skill` に `ninfer-video` と配置したSKILL.mdのpathが出ることを確認する。
OpenCode上でMCPの接続状態と3つの動画toolsが利用できることを確認し、例えば次のように依頼する：

```text
ninfer-video skillを使って C:/Videos/demo.mp4 のmetadataを確認し、
frame 100をinspect_videoで実際に見て、表示内容を説明してください。
```

frame=100には101枚以上のsource frameが必要。短い動画ならframe=0を使う。
視覚入力は同じagentの次のモデルcallへ渡る。MCP成功だけでなく、画像を見た回答が得られることを確認する。

接続失敗はport・serverログ・API keyを確認。path失敗はmountとmap、local-media-root、実ファイル名を確認。
skill未検出は起動directoryと配置先を確認。普通のOpenCode/providerで視覚入力が届かない場合は
patched exeと `opencode-ninfer` の設定を確認する。
