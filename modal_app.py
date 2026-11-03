"""vLLM on Modal: the Llama gateway model, served as an OpenAI-compatible API.

Deployed for the full run (Llama 3.1 8B Instruct on one H100). Without Modal, Ollama on a laptop
stands in (see .env.example). To deploy:

    modal secret create hf-token HF_TOKEN=...            # account with the Llama license accepted
    modal secret create llama-api-key LLAMA_API_KEY=...  # the gateway's key for this endpoint
    modal run modal_app.py::download                     # weights into the volume, once
    modal deploy modal_app.py
    # then the CLOUD block in .env: LLAMA_BASE_URL=<printed url>/v1, LLAMA_BACKEND=vllm

MODEL, GPU, LMCACHE and MODAL_APP come from the environment at deploy time (defaults: Llama
3.1 8B Instruct on an H100, app memory-gateway-llama), e.g.
    MODEL=meta-llama/Llama-3.2-3B-Instruct GPU=L4 modal deploy modal_app.py
A GPU count (GPU=H100:2) splits the model across the GPUs (tensor parallel). A second
MODAL_APP keeps another deployment up beside the default one, e.g. a 70B model:
    MODEL=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic GPU=H100:2 MODAL_APP=itara-llama-70b \
        modal deploy modal_app.py

One container at most: a second would cost a second GPU and split the prefix cache, so
requests beyond max_inputs queue instead (keep INGEST_CONCURRENCY at or below 32).

LMCACHE=1 adds LMCache (CPU offload of the KV cache). Prefix-chained hashing is kept and
CacheBlend is off: scoping by cache_salt relies on every block's key depending on the
first block. If LMCache ignores cache_salt, set LLAMA_BACKEND=ollama in .env to scope with
a prompt marker instead (it works with any prefix cache).
"""

import glob
import os
import subprocess
import threading

import modal

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
GPU = os.environ.get("GPU", "H100")
LMCACHE = os.environ.get("LMCACHE", "0") == "1"
APP = os.environ.get("MODAL_APP", "memory-gateway-llama")
TENSOR_PARALLEL = int(GPU.split(":")[1]) if ":" in GPU else 1
MAX_MODEL_LEN = 16384

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.11 doesn't cap transformers, and transformers 5 removed tokenizer methods it calls.
    .pip_install("vllm==0.11.0", "transformers>=4.56,<5", "huggingface_hub[hf_transfer]",
                 *(["lmcache"] if LMCACHE else []))
    # Containers re-import this file, so pin the deploy-time choices into the image.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models",
          "MODEL": MODEL, "LMCACHE": "1" if LMCACHE else "0", "GPU": GPU, "MODAL_APP": APP})
)
weights = modal.Volume.from_name("memory-gateway-weights", create_if_missing=True)
app = modal.App(APP, image=image)


@app.function(volumes={"/models": weights}, secrets=[modal.Secret.from_name("hf-token")],
              timeout=3600)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, ignore_patterns=["*.pth", "original/*"])
    weights.commit()


def snapshot_dir() -> str:
    """The downloaded model's directory in the volume (the newest, if downloaded twice)."""
    paths = glob.glob(f"/models/hub/models--{MODEL.replace('/', '--')}/snapshots/*")
    if not paths:
        raise RuntimeError(f"{MODEL} isn't in the volume: run `modal run modal_app.py::download`")
    return max(paths, key=os.path.getmtime)


@app.function(
    gpu=GPU,
    volumes={"/models": weights},
    secrets=[modal.Secret.from_name("llama-api-key")],
    scaledown_window=300,  # scale to zero after 5 idle minutes
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)  # a 70B model takes several minutes to load
def serve():
    # Served from the volume's directory under the repo's name: Hugging Face is never contacted
    # (the server has no token for the gated repo, and the snapshot deliberately skips
    # original/, which the hub's offline check would call incomplete).
    cmd = [
        "vllm", "serve", snapshot_dir(),
        "--served-model-name", MODEL,
        "--port", "8000",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",  # reports cached_tokens in usage
        "--api-key", os.environ["LLAMA_API_KEY"],
    ]
    if TENSOR_PARALLEL > 1:
        cmd += ["--tensor-parallel-size", str(TENSOR_PARALLEL)]
    env = dict(os.environ) | {"HF_HUB_OFFLINE": "1"}
    if LMCACHE:
        cmd += ["--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}']
        env |= {"LMCACHE_CHUNK_SIZE": "256", "LMCACHE_LOCAL_CPU": "True",
                "LMCACHE_MAX_LOCAL_CPU_SIZE": "20"}
    server = subprocess.Popen(cmd, env=env)
    # If vLLM exits (a startup error or a crash), end the container at once. Otherwise it sits
    # on the GPU until startup_timeout, and Modal starts another one: billed, and never ready.
    threading.Thread(target=lambda: os._exit(server.wait() or 1), daemon=True).start()
