# Local web UI for the Qwen-Image-2.1 kernel runtime.
#
#   python webui/server.py            # http://127.0.0.1:8189
#
# The runtime loads once at startup (FP8 DiT + FP8 text encoder + warm Triton
# kernels) and stays resident - generations then run at kernel-benchmark speeds
# (~44 s t2i @1024, ~35 s edits). Generations are serialized through a job
# queue; the browser polls per-job progress.

from __future__ import annotations

import argparse
import secrets
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "webui_output"
OUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Qwen-Image-2.1 Kernel Runtime")

RT = None
READY = threading.Event()
LOAD_ERROR: str | None = None

JOBS: dict[str, dict] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
# one generation at a time - the GPU serves a single resident model
WORKER_SEM = threading.Semaphore(1)


def _load_model() -> None:
    global RT, LOAD_ERROR
    try:
        from qwen_image_kernel.runtime import KernelRuntime, KernelRuntimeConfig

        t0 = time.perf_counter()
        rt = KernelRuntime(KernelRuntimeConfig(fp8=True, kernel_blocks=True))
        # load the text encoder + DiT now (not on the first request)
        rt.encode_prompt("warmup")
        rt.transformer = rt._build_transformer()
        # warm Triton kernels for the default 1024 shape (compiles once)
        rt.generate("warmup", output_resolution=1024, num_inference_steps=2, seed=0)
        torch.cuda.empty_cache()
        RT = rt
        READY.set()
        print(f"[site] model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as e:  # keep the server alive to show the error
        LOAD_ERROR = f"{e}\n{traceback.format_exc(limit=6)}"
        print(f"[site] MODEL LOAD FAILED: {LOAD_ERROR}", flush=True)


threading.Thread(target=_load_model, daemon=True).start()


def _worker() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        with WORKER_SEM:
            try:
                job["state"] = "running"
                job["phase"] = "encoding"
                job["t0"] = time.time()

                def cb(i: int, t):
                    job["step"] = i + 1
                    job["phase"] = "denoising"

                kwargs = dict(
                    prompt=job["prompt"],
                    output_resolution=job["size"],
                    num_inference_steps=job["steps"],
                    seed=job["seed"],
                    true_cfg_scale=1.0,
                    negative_prompt=job["negative"] or None,
                    step_callback=cb,
                )
                if job["images"]:
                    kwargs["image"] = job["images"]
                img = RT.generate(**kwargs)[0]

                name = f"{time.strftime('%Y%m%d-%H%M%S')}_{job_id[:8]}.png"
                img.save(OUT_DIR / name)
                job.update(
                    state="done", phase="done", image=f"/images/{name}",
                    elapsed_s=round(time.time() - job["t_queued"], 1),
                )
            except Exception as e:
                job.update(state="error", phase="error", error=str(e),
                           trace=traceback.format_exc(limit=4))
                print(f"[site] job {job_id} failed: {e}", flush=True)


threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    negative: str = Form(""),
    steps: int = Form(40),
    size: int = Form(1024),
    seed: int = Form(-1),
    images: list[UploadFile] = File(None),
):
    if not READY.is_set():
        return JSONResponse({"error": LOAD_ERROR or "model still loading - try again shortly"},
                            status_code=503)
    if not prompt.strip():
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    if images and len(images) > 4:
        return JSONResponse({"error": "up to 4 reference images"}, status_code=400)

    job_id = secrets.token_hex(8)
    up_dir = OUT_DIR / "uploads"
    up_dir.mkdir(exist_ok=True)
    ref_paths = []
    for up in images or []:
        if up.filename:
            p = up_dir / f"{job_id[:8]}_{secrets.token_hex(2)}.png"
            p.write_bytes(await up.read())
            ref_paths.append(str(p))

    used_seed = seed if seed >= 0 else secrets.randbelow(2**31)
    JOBS[job_id] = {
        "state": "queued", "phase": "queued", "step": 0, "steps": steps,
        "prompt": prompt, "negative": negative, "size": size, "seed": used_seed,
        "images": ref_paths, "t_queued": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "queue": JOB_QUEUE.qsize()}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {k: job.get(k) for k in
            ("state", "phase", "step", "steps", "seed", "elapsed_s", "error", "image")}


@app.get("/api/health")
def health():
    free, total = torch.cuda.mem_get_info()
    return {
        "ready": READY.is_set(),
        "load_error": LOAD_ERROR,
        "gpu": torch.cuda.get_device_name(0),
        "vram_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "vram_free_gib": round(free / 2**30, 2),
        "queue": JOB_QUEUE.qsize(),
        "jobs_done": sum(1 for j in JOBS.values() if j["state"] == "done"),
    }


app.mount("/images", StaticFiles(directory=str(OUT_DIR)), name="images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8189)
    args = ap.parse_args()
    print(f"[site] starting on http://{args.host}:{args.port} - model loads in the background", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")