# Patchforge

This repo now ships an installable CLI that wraps:

- `llama-server` for a local OpenAI-compatible endpoint
- `aider-chat` for file editing against that local model

The binary name is `patchforge`.

## Install

Install the package into a tool environment:

```bash
uv tool install .
```

Or into a project virtualenv:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

`aider-chat` is installed as a package dependency. Then bootstrap the native side:

```bash
patchforge install
```

`patchforge install` chooses the best local installation path it can find:

- Reuse an existing `llama-server` if one is already on `PATH`
- Otherwise, on macOS, prefer Homebrew and install `llama.cpp`
- After that, prefetch the default GGUF models into the llama.cpp cache

## Usage

Start the local model server in the background:

```bash
patchforge start
```

Check whether it is up:

```bash
patchforge status
```

Run Aider against that endpoint:

```bash
patchforge aider --yes-always --message "Create hello.txt with a short greeting."
```

You can also let the CLI ensure the server is running first:

```bash
patchforge aider --ensure-server --yes-always --message "Create hello.txt with a short greeting."
```

Stop the managed background server:

```bash
patchforge stop
```

Inspect the local `/v1/models` endpoint:

```bash
patchforge models
```

## Defaults

- Host: `127.0.0.1`
- Port: `8091`
- Model alias: `gemma-local`
- Default models: cached `gemma-2-9b-it` first, then cached `gemma-4-E4B-it`
- Runtime state: `.patchforge/` under the current project

## Overrides

These environment variables are supported:

```bash
LLAMA_HOST=127.0.0.1
LLAMA_PORT=8095
LLAMA_MODEL_ALIAS=my-local-model
LLAMA_MODEL_PATH=/absolute/path/to/model.gguf
LLAMA_CTX_SIZE=8192
LLAMA_PARALLEL=1
OPENAI_API_KEY=sk-local
```

You can also pass the same values as CLI flags such as `--port`, `--model-alias`, and `--model-path`.

If you want to force the Homebrew path explicitly:

```bash
patchforge install --installer brew --force-install
```
