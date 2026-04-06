from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ALIAS = "gemma-local"
DEFAULT_PORT = 8091
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CTX_SIZE = 8192
DEFAULT_PARALLEL = 1


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    quant: str
    cached_filename: str

    @property
    def hf_repo(self) -> str:
        return f"{self.repo}:{self.quant}"

    def cached_path(self, cache_dir: Path) -> Path:
        return cache_dir / self.cached_filename


DEFAULT_MODEL_SPECS = (
    ModelSpec(
        name="gemma-4-e4b-it",
        repo="ggml-org/gemma-4-E4B-it-GGUF",
        quant="Q4_K_M",
        cached_filename="ggml-org_gemma-4-E4B-it-GGUF_gemma-4-e4b-it-Q4_K_M.gguf",
    ),
    ModelSpec(
        name="gemma-2-9b-it",
        repo="second-state/gemma-2-9b-it-GGUF",
        quant="Q4_K_M",
        cached_filename="second-state_gemma-2-9b-it-GGUF_gemma-2-9b-it-Q4_K_M.gguf",
    ),
)
MODEL_SPECS_BY_NAME = {spec.name: spec for spec in DEFAULT_MODEL_SPECS}


@dataclass
class RuntimePaths:
    project_root: Path
    state_dir: Path
    logs_dir: Path
    home_dir: Path
    cache_dir: Path
    pid_file: Path
    server_state_file: Path
    server_log_file: Path


def default_llama_cache_dir() -> Path:
    env_value = os.environ.get("LLAMA_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def runtime_paths(project_root: Path) -> RuntimePaths:
    state_dir = project_root / ".patchforge"
    return RuntimePaths(
        project_root=project_root,
        state_dir=state_dir,
        logs_dir=state_dir / "logs",
        home_dir=state_dir / "home",
        cache_dir=state_dir / "cache",
        pid_file=state_dir / "llama-server.pid",
        server_state_file=state_dir / "llama-server.json",
        server_log_file=state_dir / "logs" / "llama-server.log",
    )


@dataclass
class ServerConfig:
    host: str
    port: int
    model_alias: str
    model_path: Path | None
    ctx_size: int
    parallel: int
    openai_api_key: str

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.home_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)


def default_model_paths(cache_dir: Path) -> list[Path]:
    return [spec.cached_path(cache_dir) for spec in DEFAULT_MODEL_SPECS]


def resolve_model_specs(model_names: list[str] | None, include_defaults: bool) -> list[ModelSpec]:
    ordered_names: list[str] = []
    if include_defaults:
        ordered_names.extend(spec.name for spec in DEFAULT_MODEL_SPECS)
    ordered_names.extend(model_names or [])

    resolved: list[ModelSpec] = []
    seen: set[str] = set()
    for model_name in ordered_names:
        if model_name in seen:
            continue
        try:
            spec = MODEL_SPECS_BY_NAME[model_name]
        except KeyError as exc:
            supported = ", ".join(MODEL_SPECS_BY_NAME)
            raise ToolError(f"Unknown model `{model_name}`. Supported values: {supported}") from exc
        resolved.append(spec)
        seen.add(model_name)
    return resolved


def pick_model(model_path: str | None, cache_dir: Path) -> Path:
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise ToolError(f"Model file not found: {candidate}")
        return candidate

    for candidate in default_model_paths(cache_dir):
        if candidate.is_file():
            return candidate

    raise ToolError(
        "No cached GGUF model was found. "
        "Run `patchforge install --download-default-models` or set --model-path / LLAMA_MODEL_PATH."
    )


def load_server_state(paths: RuntimePaths) -> dict[str, Any] | None:
    if not paths.server_state_file.is_file():
        return None
    try:
        return json.loads(paths.server_state_file.read_text())
    except json.JSONDecodeError:
        return None


def write_server_state(paths: RuntimePaths, state: dict[str, Any]) -> None:
    paths.server_state_file.write_text(json.dumps(state, indent=2))
    paths.pid_file.write_text(f"{state['pid']}\n")


def remove_server_state(paths: RuntimePaths) -> None:
    for target in (paths.pid_file, paths.server_state_file):
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def request_json(url: str, timeout: float = 1.0) -> dict[str, Any]:
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ToolError(str(exc)) from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON returned by {url}") from exc


def server_ready(config: ServerConfig, timeout: float = 1.0) -> bool:
    try:
        payload = request_json(f"{config.api_base}/models", timeout=timeout)
    except ToolError:
        return False

    models = payload.get("data", [])
    for item in models:
        if item.get("id") == config.model_alias:
            return True
    return False


def read_log_tail(log_file: Path, max_lines: int = 20) -> str:
    if not log_file.is_file():
        return ""
    lines = log_file.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_server(config: ServerConfig, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server_ready(config):
            return True
        time.sleep(0.5)
    return server_ready(config)


def resolve_project_root(value: str | None) -> Path:
    root = Path(value).expanduser().resolve() if value else Path.cwd().resolve()
    if not root.exists():
        raise ToolError(f"Project root does not exist: {root}")
    if not root.is_dir():
        raise ToolError(f"Project root is not a directory: {root}")
    return root


def build_server_config(args: argparse.Namespace, require_model: bool = True) -> ServerConfig:
    configured_model_path = args.model_path or os.environ.get("LLAMA_MODEL_PATH")
    model_path: Path | None
    if require_model:
        model_path = pick_model(
            model_path=configured_model_path,
            cache_dir=default_llama_cache_dir(),
        )
    elif configured_model_path:
        candidate = Path(configured_model_path).expanduser()
        model_path = candidate if candidate.is_file() else None
    else:
        model_path = next(
            (candidate for candidate in default_model_paths(default_llama_cache_dir()) if candidate.is_file()),
            None,
        )
    return ServerConfig(
        host=args.host,
        port=args.port,
        model_alias=args.model_alias,
        model_path=model_path,
        ctx_size=args.ctx_size,
        parallel=args.parallel,
        openai_api_key=args.openai_api_key,
    )


def check_llama_server_available() -> str:
    binary = shutil.which("llama-server")
    if binary:
        return binary
    raise ToolError(
        "`llama-server` was not found on PATH. "
        "Install llama.cpp and retry, or expose `llama-server` on PATH."
    )


def check_llama_cli_available() -> str:
    binary = shutil.which("llama-cli")
    if binary:
        return binary
    raise ToolError(
        "`llama-cli` was not found on PATH. "
        "Install llama.cpp first so the model bootstrap step can fetch GGUF files."
    )


def check_brew_available() -> str:
    binary = shutil.which("brew")
    if binary:
        return binary
    raise ToolError(
        "`brew` was not found on PATH. "
        "Homebrew is required for `patchforge install` to install llama.cpp."
    )


def choose_llama_cpp_installer(requested: str) -> str:
    if requested == "brew":
        check_brew_available()
        return "brew"

    existing = shutil.which("llama-server")
    if existing:
        return "existing"

    if sys.platform == "darwin" and shutil.which("brew"):
        return "brew"

    if shutil.which("brew"):
        return "brew"

    raise ToolError(
        "Could not find a supported automatic installer for llama.cpp. "
        "Install Homebrew or put `llama-server` on PATH, then retry."
    )


def build_server_command(binary: str, config: ServerConfig) -> list[str]:
    if config.model_path is None:
        raise ToolError("No model file is available for llama-server.")
    return [
        binary,
        "--offline",
        "--model",
        str(config.model_path),
        "--alias",
        config.model_alias,
        "--ctx-size",
        str(config.ctx_size),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--parallel",
        str(config.parallel),
        "--no-webui",
    ]


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ToolError(
            f"Command failed: {' '.join(command)}\n"
            f"Output:\n{completed.stdout.strip() or '(no output)'}"
        )


def brew_has_formula(brew_binary: str, formula: str) -> bool:
    completed = subprocess.run(
        [brew_binary, "list", "--formula", formula],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def build_model_download_command(binary: str, spec: ModelSpec) -> list[str]:
    return [
        binary,
        "--hf-repo",
        spec.hf_repo,
        "--prompt",
        "",
        "--n-predict",
        "0",
    ]


def ensure_llama_cpp_installed(installer: str, force_install: bool, project_root: Path) -> None:
    if installer == "existing":
        binary = check_llama_server_available()
        print(f"llama.cpp is already available at {binary}.")
        return

    if installer == "brew":
        brew_binary = check_brew_available()
        if force_install or not brew_has_formula(brew_binary, "llama.cpp"):
            print("Installing llama.cpp with Homebrew...")
            run_checked([brew_binary, "install", "llama.cpp"], cwd=project_root)
        else:
            print("llama.cpp is already installed with Homebrew.")
        return

    raise ToolError(f"Unsupported installer strategy: {installer}")


def cmd_install(args: argparse.Namespace) -> int:
    project_root = resolve_project_root(args.project_root)
    ensure_runtime_dirs(runtime_paths(project_root))

    installer = choose_llama_cpp_installer(args.installer)
    ensure_llama_cpp_installed(installer, args.force_install, project_root)

    selected_specs = resolve_model_specs(args.download_model, args.download_default_models)
    if not selected_specs:
        print("Installed llama.cpp.")
        print("No models downloaded. Use --download-default-models or --download-model to cache local GGUF models.")
        return 0

    llama_cli = check_llama_cli_available()
    cache_dir = default_llama_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected_specs:
        cached_path = spec.cached_path(cache_dir)
        if cached_path.is_file():
            print(f"Model already cached: {spec.name} ({cached_path.name})")
            continue

        print(f"Downloading {spec.name} from {spec.hf_repo} ...")
        env = os.environ.copy()
        env.pop("LLAMA_OFFLINE", None)
        run_checked(build_model_download_command(llama_cli, spec), cwd=project_root, env=env)
        if not cached_path.is_file():
            raise ToolError(
                f"Model download finished but the cache file was not found: {cached_path}"
            )
        print(f"Cached {cached_path.name}")

    print("Bootstrap complete.")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    project_root = resolve_project_root(args.project_root)
    paths = runtime_paths(project_root)
    ensure_runtime_dirs(paths)
    config = build_server_config(args)
    binary = check_llama_server_available()

    if server_ready(config):
        print(f"Server already reachable at {config.api_base} with model {config.model_alias}.")
        state = load_server_state(paths) or {}
        if "pid" not in state:
            write_server_state(
                paths,
                {
                    "pid": 0,
                    "managed": False,
                    "started_at": time.time(),
                    "server": asdict(config) | {"model_path": str(config.model_path)},
                    "log_file": str(paths.server_log_file),
                },
            )
        return 0

    if port_open(config.host, config.port):
        raise ToolError(
            f"Port {config.port} is already in use on {config.host}, "
            "but the expected llama-server endpoint is not responding."
        )

    command = build_server_command(binary, config)
    if args.foreground:
        print("Starting llama-server in the foreground.")
        print(" ".join(command))
        completed = subprocess.run(command, cwd=project_root)
        return completed.returncode

    with paths.server_log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {' '.join(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    state = {
        "pid": process.pid,
        "managed": True,
        "started_at": time.time(),
        "server": asdict(config) | {"model_path": str(config.model_path)},
        "log_file": str(paths.server_log_file),
    }
    write_server_state(paths, state)

    if wait_for_server(config, args.wait_timeout):
        print(f"Server started in the background at {config.api_base}.")
        print(f"PID: {process.pid}")
        print(f"Log: {paths.server_log_file}")
        return 0

    if process.poll() is None:
        stop_process_group(process.pid)
    remove_server_state(paths)
    log_tail = read_log_tail(paths.server_log_file)
    raise ToolError(
        "llama-server did not become ready before the timeout.\n"
        f"Log tail:\n{log_tail or '(log file is empty)'}"
    )


def stop_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not process_is_running(pid):
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def cmd_stop(args: argparse.Namespace) -> int:
    project_root = resolve_project_root(args.project_root)
    paths = runtime_paths(project_root)
    state = load_server_state(paths)
    if not state:
        print("No managed server state found.")
        return 0

    pid = int(state.get("pid") or 0)
    if pid > 0 and process_is_running(pid):
        stop_process_group(pid)
        print(f"Stopped llama-server process group {pid}.")
    else:
        print("Managed server state was stale; cleaned it up.")
    remove_server_state(paths)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project_root = resolve_project_root(args.project_root)
    paths = runtime_paths(project_root)
    state = load_server_state(paths)
    if not state:
        print("stopped")
        return 0

    pid = int(state.get("pid") or 0)
    server = state.get("server", {})
    config = ServerConfig(
        host=server.get("host", DEFAULT_HOST),
        port=int(server.get("port", DEFAULT_PORT)),
        model_alias=server.get("model_alias", DEFAULT_MODEL_ALIAS),
        model_path=Path(server.get("model_path", "")),
        ctx_size=int(server.get("ctx_size", DEFAULT_CTX_SIZE)),
        parallel=int(server.get("parallel", DEFAULT_PARALLEL)),
        openai_api_key=server.get("openai_api_key", "sk-local"),
    )
    healthy = server_ready(config)
    running = pid > 0 and process_is_running(pid)

    if healthy:
        print(f"running {config.api_base} model={config.model_alias} pid={pid or 'unmanaged'}")
        return 0

    if running:
        print(f"starting pid={pid} log={paths.server_log_file}")
        return 0

    print("stopped")
    return 1


def is_git_repo(project_root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def build_aider_command(config: ServerConfig, extra_args: list[str], use_git: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "aider.main",
        "--model",
        f"openai/{config.model_alias}",
        "--openai-api-base",
        config.api_base,
        "--openai-api-key",
        config.openai_api_key,
        "--edit-format",
        "whole",
        "--no-show-model-warnings",
        "--no-check-model-accepts-settings",
        "--no-analytics",
        "--no-check-update",
        "--no-show-release-notes",
    ]
    if not use_git:
        command.append("--no-git")
    command.extend(extra_args)
    return command


def cmd_aider(args: argparse.Namespace, extra_args: list[str]) -> int:
    project_root = resolve_project_root(args.project_root)
    paths = runtime_paths(project_root)
    ensure_runtime_dirs(paths)
    config = build_server_config(args, require_model=args.ensure_server)

    if args.ensure_server and not server_ready(config):
        start_args = argparse.Namespace(
            project_root=str(project_root),
            model_path=args.model_path,
            model_alias=args.model_alias,
            host=args.host,
            port=args.port,
            ctx_size=args.ctx_size,
            parallel=args.parallel,
            openai_api_key=args.openai_api_key,
            foreground=False,
            wait_timeout=args.wait_timeout,
        )
        cmd_start(start_args)

    command = build_aider_command(config, extra_args, use_git=is_git_repo(project_root))
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(paths.home_dir),
            "XDG_CACHE_HOME": str(paths.cache_dir),
            "AIDER_ANALYTICS": "false",
            "AIDER_CHECK_UPDATE": "false",
            "AIDER_DISABLE_PLAYWRIGHT": "true",
            "AIDER_SHOW_RELEASE_NOTES": "false",
        }
    )
    completed = subprocess.run(command, cwd=project_root, env=env)
    return completed.returncode


def cmd_models(args: argparse.Namespace) -> int:
    config = build_server_config(args, require_model=False)
    payload = request_json(f"{config.api_base}/models", timeout=2.0)
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patchforge",
        description="Run llama-server in the background and use Aider against the local endpoint.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_connection_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--project-root", default=None, help="Project directory to operate in.")
        target.add_argument("--host", default=os.environ.get("LLAMA_HOST", DEFAULT_HOST))
        target.add_argument(
            "--port",
            default=int(os.environ.get("LLAMA_PORT", DEFAULT_PORT)),
            type=int,
        )
        target.add_argument(
            "--model-alias",
            default=os.environ.get("LLAMA_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
        )
        target.add_argument("--model-path", default=None)
        target.add_argument(
            "--ctx-size",
            default=int(os.environ.get("LLAMA_CTX_SIZE", DEFAULT_CTX_SIZE)),
            type=int,
        )
        target.add_argument(
            "--parallel",
            default=int(os.environ.get("LLAMA_PARALLEL", DEFAULT_PARALLEL)),
            type=int,
        )
        target.add_argument(
            "--openai-api-key",
            default=os.environ.get("OPENAI_API_KEY", "sk-local"),
        )

    start_parser = subparsers.add_parser("start", help="Start llama-server.")
    add_connection_options(start_parser)
    start_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run llama-server in the foreground instead of daemonizing it.",
    )
    start_parser.add_argument(
        "--wait-timeout",
        default=60.0,
        type=float,
        help="Seconds to wait for the endpoint to become ready.",
    )

    stop_parser = subparsers.add_parser("stop", help="Stop the managed llama-server.")
    stop_parser.add_argument("--project-root", default=None, help="Project directory to operate in.")

    status_parser = subparsers.add_parser("status", help="Show managed llama-server status.")
    status_parser.add_argument("--project-root", default=None, help="Project directory to operate in.")

    install_parser = subparsers.add_parser(
        "install",
        help="Install llama.cpp using the preferred local strategy. GGUF model downloads are opt-in.",
    )
    install_parser.add_argument("--project-root", default=None, help="Project directory to operate in.")
    install_parser.add_argument(
        "--installer",
        choices=("auto", "brew"),
        default="auto",
        help="Installation strategy for llama.cpp. `auto` prefers an existing binary and otherwise uses Homebrew.",
    )
    install_parser.add_argument(
        "--force-install",
        action="store_true",
        help="Run the selected llama.cpp installer even if it looks already installed.",
    )
    install_parser.add_argument(
        "--download-default-models",
        action="store_true",
        help="Download the curated local GGUF defaults after installing llama.cpp.",
    )
    install_parser.add_argument(
        "--download-model",
        action="append",
        choices=tuple(MODEL_SPECS_BY_NAME),
        default=[],
        help="Download a specific local GGUF model. Repeat to fetch more than one.",
    )

    aider_parser = subparsers.add_parser("aider", help="Run Aider against the local endpoint.")
    add_connection_options(aider_parser)
    aider_parser.add_argument(
        "--ensure-server",
        action="store_true",
        help="Start the managed llama-server if the endpoint is not already reachable.",
    )
    aider_parser.add_argument(
        "--wait-timeout",
        default=60.0,
        type=float,
        help="Seconds to wait for the endpoint to become ready when --ensure-server is used.",
    )

    models_parser = subparsers.add_parser("models", help="Query the local OpenAI-compatible models endpoint.")
    add_connection_options(models_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0] == "aider":
        aider_parser = build_parser()._subparsers._group_actions[0].choices["aider"]
        args, extra_args = aider_parser.parse_known_args(argv[1:])
        args.command = "aider"
    else:
        args = parser.parse_args(argv)
        extra_args = []

    try:
        if args.command == "install":
            return cmd_install(args)
        if args.command == "start":
            return cmd_start(args)
        if args.command == "stop":
            return cmd_stop(args)
        if args.command == "status":
            return cmd_status(args)
        if args.command == "aider":
            return cmd_aider(args, extra_args)
        if args.command == "models":
            return cmd_models(args)
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
