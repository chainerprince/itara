"""vLLM on Modal: the Llama gateway model, served as an OpenAI-compatible API.

NOT YET DEPLOYED: written ahead of the Modal key. Until then Ollama on the laptop stands in
(see .env.example). To switch:

    modal secret create hf-token HF_TOKEN=...            # account with the Llama license accepted
    modal secret create llama-api-key LLAMA_API_KEY=...  # the gateway's key for this endpoint
    modal run modal_app.py::download                     # weights into the volume, once
    modal deploy modal_app.py
    # then the CLOUD block in .env: LLAMA_BASE_URL=<printed url>/v1, LLAMA_BACKEND=vllm

MODEL, GPU and LMCACHE come from the environment at deploy time, e.g.
    MODEL=meta-llama/Llama-3.1-8B-Instruct GPU=L40S modal deploy modal_app.py

LMCACHE=1 adds LMCache (CPU offload of the KV cache). Prefix-chained hashing is kept and
CacheBlend is off: scoping by cache_salt relies on every block's key depending on the
first block. If LMCache ignores cache_salt, set LLAMA_BACKEND=ollama in .env to scope with
a prompt marker instead (it works with any prefix cache).
"""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-3B-Instruct")
GPU = os.environ.get("GPU", "L4")
LMCACHE = os.environ.get("LMCACHE", "0") == "1"
MAX_MODEL_LEN = 16384

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]", *(["lmcache"] if LMCACHE else []))
    # Containers re-import this file, so pin the deploy-time choices into the image.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models",
          "MODEL": MODEL, "LMCACHE": "1" if LMCACHE else "0"})
)
weights = modal.Volume.from_name("memory-gateway-weights", create_if_missing=True)
app = modal.App("memory-gateway-llama", image=image)


@app.function(volumes={"/models": weights}, secrets=[modal.Secret.from_name("hf-token")],
              timeout=3600)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, ignore_patterns=["*.pth", "original/*"])
    weights.commit()


@app.function(
    gpu=GPU,
    volumes={"/models": weights},
    secrets=[modal.Secret.from_name("llama-api-key")],
    scaledown_window=300,  # scale to zero after 5 idle minutes
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--port", "8000",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",  # reports cached_tokens in usage
        "--api-key", os.environ["LLAMA_API_KEY"],
    ]
    env = dict(os.environ)
    if LMCACHE:
        cmd += ["--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}']
        env |= {"LMCACHE_CHUNK_SIZE": "256", "LMCACHE_LOCAL_CPU": "True",
                "LMCACHE_MAX_LOCAL_CPU_SIZE": "20"}
    subprocess.Popen(cmd, env=env)
