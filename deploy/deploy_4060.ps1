# RTX 4060 (Ada sm_89, 8 GB) deployment for the qwen-image-kernel runtime.
# Assumes: repo cloned to C:\Users\gomez\Qwen_Image, .venv created (Python 3.12).
# Logs to deploy_log.txt (plain text), fail-fast per stage with STAGE/OK/FAIL markers.
$ErrorActionPreference = "Continue"
$root = "C:\Users\gomez\Qwen_Image"
$log = "$root\deploy_log.txt"
Set-Location $root
Import-Module Microsoft.PowerShell.Utility -ErrorAction SilentlyContinue

function Stage($name) {
    Add-Content $log "`n===== STAGE $name $(Get-Date -Format 'HH:mm:ss') ====="
}
function Ok($msg) { Add-Content $log "OK: $msg"; Write-Host "OK: $msg" }
function Fail($msg) { Add-Content $log "FAIL: $msg"; Write-Host "**FAIL**: $msg"; exit 1 }

$py = "$root\.venv\Scripts\python.exe"
$pyq = $py.Replace('\', '/')

Stage "deps"
& $py -m pip install -r "$root\requirements.txt" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "pip install" }
& $py -c "import torch, diffusers, transformers, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "import check" }
Ok "deps installed"

Stage "fp8-probe-sm89"
& $py "$root\tools\probe_fp8.py" *>> $log
Ok "probe done (see log for scaling support on sm_89)"

Stage "model-download"
& "$root\.venv\Scripts\hf.exe" download Qwen/Qwen-Image-2.1 *>> $log
if ($LASTEXITCODE -ne 0) { Fail "model download" }
Ok "model downloaded"

Stage "fp8-caches"
& $py -c "import sys; sys.path.insert(0, '$root'); from qwen_image_kernel.quantize_offline import build_transformer_fp8_cache, build_text_encoder_fp8_cache; build_transformer_fp8_cache('Qwen/Qwen-Image-2.1', '$root/fp8_cache/transformer_fp8.safetensors'); build_text_encoder_fp8_cache('Qwen/Qwen-Image-2.1', '$root/fp8_cache/text_encoder_fp8.safetensors')" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "fp8 cache build" }
Ok "fp8 caches built"

Stage "tests"
& $py "$root\tests\test_attention.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "attention tests" }
& $py "$root\tests\test_kv_cache.py" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "kv cache tests" }
Ok "kernel tests pass on sm_89"

Stage "first-generation-512"
& $py "$root\generate.py" --prompt "a ceramic teapot on a wooden table, morning light, photorealistic" --steps 40 --seed 42 --size 512 --out "$root\4060_512.png" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "first generation" }
Ok "512 generation done"

Stage "generation-768"
& $py "$root\generate.py" --prompt "a ceramic teapot on a wooden table, morning light, photorealistic" --steps 40 --seed 42 --size 768 --out "$root\4060_768.png" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "768 generation" }
Ok "768 generation done"

Add-Content $log "`n===== DEPLOYMENT COMPLETE $(Get-Date -Format 'HH:mm:ss') ====="
Write-Host "DEPLOYMENT COMPLETE"