from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from patchforge.cli import ensure_runtime_dirs
from patchforge.cli import main
from patchforge.cli import runtime_paths
from patchforge.cli import write_server_state


class CliCommandSmokeTests(unittest.TestCase):
    def test_install_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            stdout = io.StringIO()
            with mock.patch("patchforge.cli.choose_llama_cpp_installer", return_value="existing") as choose_mock:
                with mock.patch("patchforge.cli.ensure_llama_cpp_installed") as ensure_mock:
                    with contextlib.redirect_stdout(stdout):
                        exit_code = main(["install", "--project-root", str(root), "--skip-models"])

            self.assertEqual(exit_code, 0)
            choose_mock.assert_called_once_with("auto")
            ensure_mock.assert_called_once_with("existing", False, root)
            self.assertTrue((root / ".patchforge").is_dir())
            self.assertIn("Skipped model downloads.", stdout.getvalue())

    def test_start_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model_path = root / "demo.gguf"
            model_path.write_text("model")
            process = mock.Mock(pid=4321)
            stdout = io.StringIO()

            with mock.patch("patchforge.cli.check_llama_server_available", return_value="/usr/bin/llama-server"):
                with mock.patch("patchforge.cli.server_ready", return_value=False):
                    with mock.patch("patchforge.cli.port_open", return_value=False):
                        with mock.patch("patchforge.cli.wait_for_server", return_value=True):
                            with mock.patch("patchforge.cli.subprocess.Popen", return_value=process) as popen_mock:
                                with contextlib.redirect_stdout(stdout):
                                    exit_code = main(
                                        [
                                            "start",
                                            "--project-root",
                                            str(root),
                                            "--model-path",
                                            str(model_path),
                                            "--wait-timeout",
                                            "0.01",
                                        ]
                                    )

            self.assertEqual(exit_code, 0)
            command = popen_mock.call_args.args[0]
            self.assertEqual(command[0], "/usr/bin/llama-server")
            self.assertIn("--model", command)
            self.assertIn(str(model_path), command)

            state = runtime_paths(root).server_state_file.read_text()
            self.assertIn('"pid": 4321', state)
            self.assertIn("Server started in the background", stdout.getvalue())

    def test_stop_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = runtime_paths(root)
            ensure_runtime_dirs(paths)
            write_server_state(
                paths,
                {
                    "pid": 4321,
                    "managed": True,
                    "started_at": 0.0,
                    "server": {},
                    "log_file": str(paths.server_log_file),
                },
            )

            stdout = io.StringIO()
            with mock.patch("patchforge.cli.process_is_running", return_value=True):
                with mock.patch("patchforge.cli.stop_process_group") as stop_mock:
                    with contextlib.redirect_stdout(stdout):
                        exit_code = main(["stop", "--project-root", str(root)])

        self.assertEqual(exit_code, 0)
        stop_mock.assert_called_once_with(4321)
        self.assertFalse(paths.server_state_file.exists())
        self.assertIn("Stopped llama-server process group 4321.", stdout.getvalue())

    def test_status_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["status", "--project-root", str(root)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "stopped")

    def test_aider_command_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            completed = mock.Mock(returncode=0)

            with mock.patch("patchforge.cli.is_git_repo", return_value=False):
                with mock.patch("patchforge.cli.subprocess.run", return_value=completed) as run_mock:
                    exit_code = main(["aider", "--project-root", str(root), "--message", "hi"])

        self.assertEqual(exit_code, 0)
        command = run_mock.call_args.args[0]
        env = run_mock.call_args.kwargs["env"]

        self.assertIn("-m", command)
        self.assertIn("aider.main", command)
        self.assertIn("--no-git", command)
        self.assertEqual(command[-2:], ["--message", "hi"])
        self.assertEqual(run_mock.call_args.kwargs["cwd"], root)
        self.assertEqual(env["HOME"], str(runtime_paths(root).home_dir))
        self.assertEqual(env["XDG_CACHE_HOME"], str(runtime_paths(root).cache_dir))

    def test_models_command_smoke(self) -> None:
        stdout = io.StringIO()
        with mock.patch(
            "patchforge.cli.request_json",
            return_value={"data": [{"id": "gemma-local"}]},
        ):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["models"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"id": "gemma-local"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
