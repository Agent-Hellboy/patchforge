from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from patchforge.cli import DEFAULT_MODEL_ALIAS
from patchforge.cli import DEFAULT_MODEL_SPECS
from patchforge.cli import ServerConfig
from patchforge.cli import build_aider_command
from patchforge.cli import build_model_download_command
from patchforge.cli import choose_llama_cpp_installer
from patchforge.cli import pick_model
from patchforge.cli import runtime_paths


class CliTests(unittest.TestCase):
    def test_pick_model_prefers_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            explicit = cache_dir / "custom.gguf"
            explicit.write_text("x")
            self.assertEqual(pick_model(str(explicit), cache_dir), explicit)

    def test_pick_model_uses_default_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            default = cache_dir / "second-state_gemma-2-9b-it-GGUF_gemma-2-9b-it-Q4_K_M.gguf"
            default.write_text("x")
            self.assertEqual(pick_model(None, cache_dir), default)

    def test_runtime_paths_are_project_scoped(self) -> None:
        root = Path("/tmp/demo-project")
        paths = runtime_paths(root)
        self.assertEqual(paths.state_dir, root / ".patchforge")
        self.assertEqual(paths.server_log_file, root / ".patchforge" / "logs" / "llama-server.log")

    def test_build_aider_command_uses_local_endpoint(self) -> None:
        config = ServerConfig(
            host="127.0.0.1",
            port=8091,
            model_alias=DEFAULT_MODEL_ALIAS,
            model_path=Path("/tmp/model.gguf"),
            ctx_size=8192,
            parallel=1,
            openai_api_key="sk-local",
        )
        command = build_aider_command(config, ["--message", "hi"], use_git=False)
        self.assertIn("--no-git", command)
        self.assertIn("openai/gemma-local", command)
        self.assertEqual(command[-2:], ["--message", "hi"])

    def test_choose_llama_cpp_installer_prefers_existing_binary(self) -> None:
        with mock.patch("patchforge.cli.shutil.which") as which_mock:
            which_mock.side_effect = lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None
            self.assertEqual(choose_llama_cpp_installer("auto"), "existing")

    def test_build_model_download_command_uses_hf_repo(self) -> None:
        command = build_model_download_command("/opt/homebrew/bin/llama-cli", DEFAULT_MODEL_SPECS[0])
        self.assertEqual(
            command,
            [
                "/opt/homebrew/bin/llama-cli",
                "--hf-repo",
                "second-state/gemma-2-9b-it-GGUF:Q4_K_M",
                "--prompt",
                "",
                "--n-predict",
                "0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
