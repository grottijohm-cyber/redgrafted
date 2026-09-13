import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import comfy_launcher
import configure_startup
import runtime_health
import worker


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.proc = self.root / "proc"
        self.cgroup = self.root / "cgroup"
        (self.proc / "self").mkdir(parents=True)
        self.cgroup.mkdir()
        (self.proc / "meminfo").write_text("MemTotal: 268435456 kB\nMemAvailable: 209715200 kB\n")
        self.enterContext(patch.object(runtime_health, "PROC_ROOT", self.proc))
        self.enterContext(patch.object(runtime_health, "CGROUP_ROOT", self.cgroup))

    def test_reports_container_allowance_instead_of_host_ram(self):
        (self.proc / "self/cgroup").write_text("0::/\n")
        (self.cgroup / "memory.max").write_text(str(32 * 1024**3))
        (self.cgroup / "memory.current").write_text(str(29 * 1024**3))
        (self.cgroup / "memory.events").write_text("oom 2\noom_kill 1\n")
        data = runtime_health.memory_snapshot()
        self.assertEqual(data["host_total_bytes"], 256 * 1024**3)
        self.assertEqual(data["effective_total_bytes"], 32 * 1024**3)
        self.assertEqual(data["container_used_bytes"], 29 * 1024**3)
        self.assertEqual(data["oom_kills"], 1)

    def test_inherits_a_tighter_parent_container_limit(self):
        (self.proc / "self/cgroup").write_text("0::/worker/child\n")
        leaf = self.cgroup / "worker/child"
        leaf.mkdir(parents=True)
        (leaf / "memory.max").write_text("max")
        (leaf / "memory.current").write_text("100")
        (leaf.parent / "memory.max").write_text("4096")
        (leaf.parent / "memory.current").write_text("2048")
        data = runtime_health.memory_snapshot()
        self.assertEqual(data["container_limit_bytes"], 4096)
        self.assertEqual(data["container_used_bytes"], 2048)

    def test_v1_limits_without_falsely_labeling_failcnt_as_a_kill(self):
        (self.proc / "self/cgroup").write_text("5:memory:/docker/worker\n")
        leaf = self.cgroup / "memory/docker/worker"
        leaf.mkdir(parents=True)
        (leaf / "memory.limit_in_bytes").write_text("8192")
        (leaf / "memory.usage_in_bytes").write_text("4096")
        (leaf / "memory.failcnt").write_text("100")
        data = runtime_health.memory_snapshot()
        self.assertEqual(data["container_limit_bytes"], 8192)
        self.assertIsNone(data["oom_kills"])

    def test_missing_cgroup_is_reported_as_unknown(self):
        data = runtime_health.memory_snapshot()
        self.assertIsNone(data["container_limit_bytes"])
        self.assertEqual(data["effective_total_bytes"], 256 * 1024**3)

    def test_attributes_only_new_oom_kills(self):
        before = {"oom_kills": 2, "container_limit_bytes": 32 * 1024**3}
        after = {**before, "oom_kills": 3}
        with patch.object(runtime_health, "process_status", return_value={"state": "unknown", "pid": None}), \
                patch.object(runtime_health, "memory_snapshot", side_effect=[before, after]):
            monitor = runtime_health.ComfyMonitor()
            with self.assertRaisesRegex(runtime_health.ComfyUnavailableError, "new out-of-memory.*32.0 GiB"):
                monitor.fail("ComfyUI exited")
        with patch.object(runtime_health, "process_status", return_value={"state": "unknown", "pid": None}), \
                patch.object(runtime_health, "memory_snapshot", return_value=before):
            monitor = runtime_health.ComfyMonitor()
            with self.assertRaises(runtime_health.ComfyUnavailableError) as raised:
                monitor.fail("ComfyUI exited")
            self.assertNotIn("recorded a new", str(raised.exception))


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CrashTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.enterContext(patch.object(worker.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(worker.time, "sleep", self.clock.sleep))
        self.enterContext(patch.object(runtime_health, "process_status", return_value={"state": "unknown", "pid": None}))

    def response(self, done=True):
        result = Mock()
        result.json.return_value = {"prompt-1": {"status": {"completed": True}}} if done else {}
        return result

    def test_connection_refusals_fail_after_one_minute_not_full_job_budget(self):
        with patch.object(worker.requests, "get", side_effect=worker.requests.ConnectionError("refused")):
            with self.assertRaisesRegex(runtime_health.ComfyUnavailableError, "unreachable for 60 seconds"):
                worker.wait_for_history("prompt-1", deadline=6600)
        self.assertEqual(self.clock.now, 60)

    def test_successful_connection_resets_the_failure_window(self):
        failures = [worker.requests.ConnectionError("refused")] * 20
        with patch.object(worker.requests, "get", side_effect=[*failures, self.response(False), *failures, self.response()]):
            result = worker.wait_for_history("prompt-1", deadline=6600)
        self.assertTrue(result["status"]["completed"])
        self.assertGreater(self.clock.now, 60)

    def test_slow_responses_are_not_misclassified_as_process_death(self):
        responses = [worker.requests.ReadTimeout("busy")] * 35 + [self.response()]
        with patch.object(worker.requests, "get", side_effect=responses):
            result = worker.wait_for_history("prompt-1", deadline=6600)
        self.assertTrue(result["status"]["completed"])

    def test_job_failure_retires_the_dead_worker_without_resubmitting(self):
        output = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(output, format="PNG")
        job = {"input": {"image": base64.b64encode(output.getvalue()).decode(), "prompt": "Clouds drift slowly."}}
        with patch.object(worker, "WORKFLOW_PATH", Path("api-workflow.json")), \
                patch.object(worker, "ensure_models"), patch.object(worker, "wait_for_comfyui"), \
                patch.object(worker, "upload_input_image", return_value="test.png"), \
                patch.object(worker, "queue_workflow", return_value="prompt-1") as queue, \
                patch.object(worker, "wait_for_history", side_effect=runtime_health.ComfyUnavailableError("ComfyUI exited")), \
                patch.object(worker, "cancel_workflow") as cancel:
            result = worker.handle_job(job)
        self.assertTrue(result["refresh_worker"])
        self.assertIn("ComfyUI exited", result["error"])
        queue.assert_called_once()
        cancel.assert_not_called()


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process health check")
    def test_real_exited_child_is_detected_before_any_http_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "comfyui.pid"
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait(timeout=5)
            pid_file.write_text(str(child.pid))
            with patch.object(runtime_health, "COMFY_PID_FILE", pid_file), \
                    patch.object(worker.requests, "get") as get:
                with self.assertRaisesRegex(runtime_health.ComfyUnavailableError, "ComfyUI exited"):
                    worker.wait_for_history("prompt-1")
            get.assert_not_called()

    def test_zombie_pid_and_restarted_process_cannot_keep_a_job_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "comfyui.pid"
            pid_file.write_text("1234")
            (root / "1234").mkdir()
            (root / "1234/stat").write_text("1234 (python worker) Z " + "0 " * 40)
            with patch.object(runtime_health, "COMFY_PID_FILE", pid_file), \
                    patch.object(runtime_health, "PROC_ROOT", root), \
                    patch.object(runtime_health.os, "kill"):
                self.assertEqual(runtime_health.process_status()["state"], "dead")
        with patch.object(runtime_health, "process_status", side_effect=[
                {"state": "alive", "pid": 1, "starttime": "100"},
                {"state": "alive", "pid": 1, "starttime": "200"}]):
            monitor = runtime_health.ComfyMonitor()
            with self.assertRaisesRegex(runtime_health.ComfyUnavailableError, "restarted"):
                monitor.check()


class StartupTests(unittest.TestCase):
    def test_adapter_preserves_both_launch_modes_and_pid_tracking(self):
        source = 'COMFY_PID_FILE="/tmp/comfyui.pid"\n'
        source += 'python -u /comfyui/main.py --listen &\necho $! > "$COMFY_PID_FILE"\n'
        source += 'python -u /comfyui/main.py --log-stdout &\necho $! > "$COMFY_PID_FILE"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "start.sh"
            path.write_text(source)
            configure_startup.configure(path)
            self.assertEqual(path.read_text(), source.replace("/comfyui/main.py", "/comfy_launcher.py"))
            path.write_text("python /new/upstream.py\n")
            with self.assertRaises(RuntimeError):
                configure_startup.configure(path)
            self.assertEqual(path.read_text(), "python /new/upstream.py\n")

    def test_launcher_executes_main_with_original_flags_and_same_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            fake_main = root / "main.py"
            fake_main.write_text("import json,os,sys\nfrom pathlib import Path\nPath(os.environ['LAUNCH_RESULT']).write_text(json.dumps({'pid':os.getpid(),'args':sys.argv}))\n")
            env = {**os.environ, "LAUNCH_RESULT": str(result_path), "COMFY_MEMORY_PROFILE": "conservative"}
            child = subprocess.Popen([sys.executable, "-c", "import comfy_launcher,sys; comfy_launcher.launch(sys.argv.pop(1))", str(fake_main), "--log-stdout"], env=env, stdout=subprocess.DEVNULL)
            self.assertEqual(child.wait(timeout=5), 0)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["pid"], child.pid)
            self.assertEqual(result["args"][0], str(fake_main))
            self.assertIn("--log-stdout", result["args"])
            self.assertIn("--disable-pinned-memory", result["args"])
            self.assertIn("--fast-disk", result["args"])
            self.assertNotIn("--disable-dynamic-vram", result["args"])


if __name__ == "__main__":
    unittest.main()
