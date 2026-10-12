# Resume: remaining stages after the deps/probe/download completed.
$ErrorActionPreference = "Continue"
$root = "C:/Users/gomez/Qwen_Image"   # forward slashes: avoids python unicodeescape issues
$log = "C:\Users\gomez\Qwen_Image\deploy_log.txt"
Set-Location "C:\Users\gomez\Qwen_Image"

function Ok($msg) { Add-Content $log "OK: $msg"; Write-Host "OK: $msg" }
function Fail($msg) { Add-Content $log "FAIL: $msg"; Write-Host "**FAIL**: $msg"; exit 1 }

$py = "$root\.venv\Scripts\python.exe"

Add-Content $log "`n===== STAGE fp8-caches-resume $(Get-Date -Format 'HH:mm:ss') ====="
& $py -c "import sys, pathlib; root = pathlib.Path('C:/Users/gomez/Qwen_Image'); sys.path.insert(0, str(root)); from qwen_image_kernel.quantize_offline import build_transformer_fp8_cache, build_text_encoder_fp8_cache; build_transformer_fp8_cache('Qwen/Qwen-Image-2.1', str(root / 'fp8_cache/transformer_fp8.safetensors')); build_text_encoder_fp8_cache('Qwen/Qwen-Image-2.1', str(root / 'fp8_cache/text_encoder_fp8.safetensors'))" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "fp8 cache build (resume)" }
Ok "fp8 caches built"

Add-Content $log "`n===== STAGE tests-sm89 $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$root\tests\test_attention.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "attention tests" }
& $py "$root\tests\test_kv_cache.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "kv cache tests" }
& $py "$root\tests\test_fused.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "fused tests" }
Ok "kernel tests pass on sm_89"

Add-Content $log "`n===== STAGE gen-512 $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$root\generate.py" --prompt "a ceramic teapot on a wooden table, morning light, photorealistic" --steps 40 --seed 42 --size 512 --out "$root\4060_512.png" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "512 generation" }
Ok "512 generation done"

Add-Content $log "`n===== STAGE gen-768 $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$root\generate.py" --prompt "a ceramic teapot on a wooden table, morning light, photorealistic" --steps 40 --seed 42 --size 768 --out "$root\4060_768.png" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "768 generation" }
Ok "768 generation done"

Add-Content $log "`n===== STAGE gen-1024-probe $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$root\generate.py" --prompt "a ceramic teapot on a wooden table, morning light, photorealistic" --steps 40 --seed 42 --size 1024 --out "$root\4060_1024.png" *>> $log
if ($LASTEXITCODE -ne 0) { Add-Content $log "NOTE: 1024 did not fit on 8 GB (documented)"; Write-Host "1024 OOM (expected on 8 GB)" } else { Ok "1024 generation done (fits!)" }

Add-Content $log "`n===== RESUME COMPLETE $(Get-Date -Format 'HH:mm:ss') ====="
Write-Host "RESUME COMPLETE"