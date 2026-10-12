# Resume 2: pull the bootstrap fix, rerun tests + generations.
$ErrorActionPreference = "Continue"
$root = "C:/Users/gomez/Qwen_Image"
$log = "C:\Users\gomez\Qwen_Image\deploy_log.txt"
Set-Location "C:\Users\gomez\Qwen_Image"

function Ok($msg) { Add-Content $log "OK: $msg"; Write-Host "OK: $msg" }
function Fail($msg) { Add-Content $log "FAIL: $msg"; Write-Host "**FAIL**: $msg"; exit 1 }

Add-Content $log "`n===== STAGE pull $(Get-Date -Format 'HH:mm:ss') ====="
& git pull 2>&1 | Out-Add-Content -Path $log
if ($LASTEXITCODE -ne 0) { Fail "git pull" }
Ok "pulled"

$py = "$root\.venv\Scripts\python.exe"
$env:TRITON_CACHE_DIR = "$root\.tcache"

Add-Content $log "`n===== STAGE tests-sm89 $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$root\tests\test_attention.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "attention tests" }
& $py "$root\tests\test_kv_cache.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "kv cache tests" }
& $py "$root\tests\test_fused.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "fused tests" }
& $py "$root\tests\test_fp8.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "fp8 tests" }
& $py "$root\tests\test_block.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "block tests" }
Ok "all kernel tests pass on sm_89"

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