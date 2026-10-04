"""The cluster launcher's recorded-deployment scan: `down` and `status`
without --config (or with --all) act on every deployment staged under
DGPP_LOG_DIR/deployments, not on the default deployment file. Single-node
namespaces use recorded `sleep` processes. Peer staging commands run in
local shells with file transfer stubbed out; no SSH is needed."""
import json
import io
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_process

LAUNCHER = ROOT / "scripts" / "dgpp-cluster"
LAUNCHER_MODULE = runpy.run_path(str(LAUNCHER))
Cluster = LAUNCHER_MODULE["Cluster"]


class LauncherScanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log_root = self.root / "log"
        (self.root / ".env").write_text(
            'DGPP_NODES="127.0.0.1"\n'
            f'DGPP_LOG_DIR="{self.log_root}"\n'
            f'DGPP_STAGE_DIR="{self.root}/stage"\n'
            f'DGPP_RELEASE_DIR="{self.root}/releases"\n'
        )
        self.environ = {
            "DGPP_ENV_FILE": str(self.root / ".env"),
            "DGPP_BUILD_DIR": str(self.root / "no-build"),  # status's --version probe: "?"
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": os.environ.get("HOME", str(self.root)),
        }

    def namespace(self, name, model, record_source=True):
        """A staged single-node deployment, as the launcher leaves it."""
        d = self.log_root / "deployments" / name
        d.mkdir(parents=True)
        (d / "cluster.resolved.json").write_text(json.dumps({
            "model": model, "engine": {}, "nodes": ["127.0.0.1"], "ssh_user": "tester",
            "ports": {"http": 18080, "fabric": 29970, "journal": 29971},
            "http": {"bind_host": "127.0.0.1", "port": 18080},
            "paths": {"log_dir": str(d), "stage_dir": f"{self.root}/stage/deployments/{name}",
                      "release_dir": f"{self.root}/releases"},
            "node_env": [{}],
        }))
        if record_source:
            (d / "deployment.path").write_text(f"{self.root}/{name}.json\n")
        return d

    def launch(self, d):
        """Rank 0 of the namespace: a recorded session-leader sleep."""
        proc = cluster_process.launch(d / "rank0.process.json", ["sleep", "300"], d / "serve_r0.log", d)
        self.addCleanup(self.kill, proc)
        return proc

    @staticmethod
    def kill(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

    def launcher(self, *argv):
        return subprocess.run([sys.executable, str(LAUNCHER), *argv], env=self.environ,
                              text=True, capture_output=True, timeout=120)

    def test_down_without_config_stops_every_running_deployment(self):
        running = self.namespace("aaaa", "org/running")
        also = self.namespace("bbbb", "org/also-running", record_source=False)
        self.namespace("cccc", "org/idle")
        first, second = self.launch(running), self.launch(also)
        result = self.launcher("down")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        self.assertIn(f"=== stopping org/running on 1 node(s) [aaaa] ({self.root}/aaaa.json): "
                      f"rank 0 (127.0.0.1) pid {first.pid}", out)
        self.assertIn(f"=== stopping org/also-running on 1 node(s) [bbbb] (deployment file not recorded): "
                      f"rank 0 (127.0.0.1) pid {second.pid}", out)
        self.assertIn("org/idle on 1 node(s) [cccc]", out)
        self.assertIn("): not running", out)
        self.assertIn("2 deployment(s) stopped", out)
        for proc in (first, second):
            proc.wait(timeout=10)
        for name in ("aaaa", "bbbb"):
            self.assertIsNone(cluster_process.running(self.log_root / "deployments" / name / "rank0.process.json"))

    def test_all_flag_is_the_same_scan(self):
        proc = self.launch(self.namespace("dddd", "org/model"))
        result = self.launcher("down", "--all")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"rank 0 (127.0.0.1) pid {proc.pid}", result.stdout)
        self.assertIn("1 deployment(s) stopped", result.stdout)
        proc.wait(timeout=10)

    def test_status_hides_stopped_deployments_unless_all_is_requested(self):
        live = self.namespace("eeee", "org/live")
        (live / "serve_r0.log").write_text("2026-09-12 15:31:40.001 INFO  serve: listening on :18080\n")
        proc = self.launch(live)
        stopped = self.namespace("ffff", "org/stopped")
        (stopped / "serve_r0.log").write_text(
            "2026-09-12 05:26:48.583 INFO  rank 0: memory plan total 108.38 GiB\n"
            "2026-09-12 05:26:48.665 ERROR serve: rank 0: the configuration needs 108.38 GiB plus 8.00 GiB headroom\n")
        self.namespace("gggg", "org/never-booted")
        result = self.launcher("status")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        # The running deployment: its ranks and the end of rank 0's log.
        self.assertIn(f"=== org/live on 1 node(s) [eeee] ({self.root}/eeee.json)", out)
        self.assertIn(f"rank 0 (127.0.0.1): alive ({proc.pid})", out)
        self.assertIn("  2026-09-12 15:31:40.001 INFO  serve: listening on :18080", out)
        self.assertNotIn("org/stopped", out)
        self.assertNotIn("org/never-booted", out)
        self.assertNotIn("no deployments running", out)
        self.assertNotIn("not running", out)
        result = self.launcher("status", "--all")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        self.assertIn(f"rank 0 (127.0.0.1): alive ({proc.pid})", out)
        # Stopped ones: one line each, saying how they ended; no rank listing.
        self.assertIn(f"org/stopped on 1 node(s) [ffff] ({self.root}/ffff.json): not running "
                      "(rank 0's log ends 2026-09-12 05:26:48: ERROR serve: rank 0: the configuration needs", out)
        self.assertIn("org/never-booted on 1 node(s) [gggg]", out)
        self.assertIn(": not running (no rank 0 log)", out)
        self.assertNotIn("=== org/stopped", out)
        self.assertNotIn("DOWN", out)
        self.assertIsNotNone(cluster_process.running(self.log_root / "deployments/eeee/rank0.process.json"))

    def test_nothing_recorded_is_not_an_error(self):
        result = self.launcher("status")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(result.stdout.strip().endswith("no deployments running"))
        for argv in (("down",), ("status", "--all")):
            result = self.launcher(*argv)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"no deployment recorded under {self.log_root}/deployments", result.stdout)
        self.namespace("gggg", "org/idle")
        result = self.launcher("down")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no deployment was running; nothing stopped", result.stdout)

    def test_status_reports_no_running_deployments_with_stale_records(self):
        stopped = self.namespace("aaaa", "org/stopped")
        proc = self.launch(stopped)
        self.kill(proc)
        self.namespace("bbbb", "org/never-booted")
        result = self.launcher("status")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertTrue(result.stdout.strip().endswith("no deployments running"))
        # Status keeps records and logs available for explicit inspection.
        self.assertTrue((stopped / "rank0.process.json").is_file())
        self.assertTrue((stopped / "serve_r0.log").is_file())

    def test_all_rejects_a_config_a_log_dir_and_other_verbs(self):
        config = self.root / "x.json"
        config.write_text(json.dumps({"model": "org/model", "world_size": 1}))
        for argv, message in ((("down", "--all", "--config", str(config)), "--all takes no --config"),
                              (("down", "--log-dir", str(self.root / "elsewhere")), "--all scans"),
                              (("up", "--all", "--config", str(config)), "--all applies to down and status")):
            result = self.launcher(*argv)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(message, result.stderr)

    def test_down_with_config_names_a_deployment_that_is_not_running(self):
        config = self.root / "never.json"
        config.write_text(json.dumps({"model": "org/never", "world_size": 1}))
        result = self.launcher("down", "--config", str(config))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("=== org/never on 1 node(s) [", result.stdout)
        self.assertIn("no rank recorded as running; nothing to stop", result.stdout)
        self.assertNotIn("op-stream identity", result.stdout)  # never staged: nothing to collect

    def cluster(self, namespace, peers=0):
        config_path = namespace / "cluster.resolved.json"
        config = json.loads(config_path.read_text())
        config["nodes"] += [f"peer{rank}" for rank in range(1, peers + 1)]
        config_path.write_text(json.dumps(config))
        with patch.dict(os.environ, self.environ, clear=True):
            return Cluster(SimpleNamespace(), namespace=str(namespace))

    def test_binary_selection_defaults_to_release_with_explicit_testing_override(self):
        namespace = self.namespace("selection", "org/model")
        env = {key: value for key, value in self.environ.items() if key != "DGPP_BUILD_DIR"}
        cases = (
            ({}, {}, ROOT / "build-release/6ix-Serve", False),
            ({"DGPP_BUILD_DIR": "build-ci"}, {}, ROOT / "build-ci/6ix-Serve", False),
            ({}, {"release": "0.1.0"}, self.root / "releases/dgpp-0.1.0/bin/dgpp-serve", True),
            ({"DGPP_BUILD_DIR": "build-ci"}, {"release": "0.1.0"},
             self.root / "releases/dgpp-0.1.0/bin/dgpp-serve", True),
            ({}, {"release": "0.1.0", "bin": str(ROOT / "build-ci/dgpp-serve")},
             ROOT / "build-ci/dgpp-serve", False),
        )
        for overrides, arguments, expected, installed in cases:
            with self.subTest(overrides=overrides, arguments=arguments), \
                    patch.dict(os.environ, {**env, **overrides}, clear=True):
                cluster = Cluster(SimpleNamespace(**arguments), namespace=str(namespace))
                self.assertEqual(cluster.bin, str(expected))
                self.assertEqual(cluster.installed, installed)

    def test_unknown_peer_is_failure_and_scan_still_stops_other_deployment(self):
        unknown = self.namespace("aaaa", "org/unknown")
        self.cluster(unknown, peers=1)
        proc = self.launch(self.namespace("bbbb", "org/reachable"))
        for verb in ("status", "down"):
            with self.subTest(verb=verb), patch.dict(os.environ, self.environ, clear=True), \
                    patch.object(Cluster, "remote_control", side_effect=RuntimeError("SSH refused")), \
                    redirect_stdout(io.StringIO()) as output:
                result = LAUNCHER_MODULE["every_deployment"](SimpleNamespace(), verb)
            self.assertNotEqual(result, 0)
            self.assertIn("UNKNOWN", output.getvalue())
            self.assertIn("SSH refused", output.getvalue())
            self.assertNotIn("no deployment was running", output.getvalue())
        proc.wait(timeout=10)
        self.assertIn("1 deployment(s) stopped", output.getvalue())
        # With the reachable deployment stopped, UNKNOWN must still be visible.
        with patch.dict(os.environ, self.environ, clear=True), \
                patch.object(Cluster, "remote_control", side_effect=RuntimeError("SSH refused")), \
                redirect_stdout(io.StringIO()) as output:
            result = LAUNCHER_MODULE["every_deployment"](SimpleNamespace(), "status")
        self.assertNotEqual(result, 0)
        self.assertIn("UNKNOWN", output.getvalue())
        self.assertNotIn("no deployments running", output.getvalue())

    def test_corrupt_record_does_not_abort_scan_of_other_deployments(self):
        invalid = self.namespace("aaaa", "org/invalid")
        (invalid / "rank0.process.json").write_text("{")
        proc = self.launch(self.namespace("bbbb", "org/reachable"))
        result = self.launcher("down", "--all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UNKNOWN", result.stdout)
        self.assertIn("1 deployment(s) stopped", result.stdout)
        proc.wait(timeout=10)

    def test_corrupt_config_does_not_abort_scan_of_other_deployments(self):
        invalid = self.namespace("aaaa", "org/invalid")
        (invalid / "cluster.resolved.json").write_text("{")
        proc = self.launch(self.namespace("bbbb", "org/reachable"))
        result = self.launcher("down", "--all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1 deployment(s) stopped", result.stdout)
        proc.wait(timeout=10)

    def test_unreachable_peer_does_not_prevent_reachable_peer_cleanup(self):
        namespace = self.namespace("aaaa", "org/peers")
        cluster = self.cluster(namespace, peers=2)
        peer_namespace = self.namespace("peer-process", "org/test-process")
        proc = self.launch(peer_namespace)
        state = peer_namespace / "rank0.process.json"

        def remote(action, rank, host, extra=()):
            if rank == 1:
                raise RuntimeError("SSH refused")
            if action == "status":
                return cluster_process.running(state)
            cluster_process.send_signal(state, getattr(signal, "SIG" + extra[-1]))

        wait = getattr(cluster, "wait_ranks", None)
        with patch.object(cluster, "remote_control", side_effect=remote), \
                patch.object(cluster, "wait_ranks", create=True,
                             side_effect=lambda ranks, seconds: wait(ranks, min(seconds, 2))), \
                redirect_stdout(io.StringIO()) as output:
            result = cluster.down()
        self.assertNotEqual(result, 0)
        self.assertIn("UNKNOWN", output.getvalue())
        proc.wait(timeout=10)

    def test_force_kill_is_verified_before_reporting_success(self):
        namespace = self.namespace("aaaa", "org/stuck")
        proc = self.launch(namespace)
        cluster = self.cluster(namespace)
        with patch.object(cluster, "signal_rank"), \
                patch.object(cluster, "wait_ranks", create=True,
                             side_effect=lambda ranks, seconds: cluster.live_ranks()), \
                patch("time.sleep"), redirect_stdout(io.StringIO()) as output:
            result = cluster.down()
        self.assertNotEqual(result, 0)
        self.assertIn("still running", output.getvalue())
        self.assertIsNone(proc.poll())

    def test_up_replace_does_not_stage_after_unconfirmed_shutdown(self):
        namespace = self.namespace("aaaa", "org/stuck")
        self.launch(namespace)
        cluster = self.cluster(namespace)
        cluster.replace = True
        with patch.object(cluster, "_down", create=True, return_value=1), \
                patch.object(cluster, "down", return_value=1), \
                patch.object(cluster, "stage", return_value=False) as stage:
            cluster.skip_preflight = True
            self.assertNotEqual(cluster.up(), 0)
            stage.assert_not_called()

    def test_down_refuses_concurrent_launcher(self):
        import fcntl
        namespace = self.namespace("aaaa", "org/locked")
        proc = self.launch(namespace)
        with (namespace / "launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.launcher("down", "--all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another launcher", result.stdout + result.stderr)
        self.assertIsNone(proc.poll())

    def check_peer_staging(self, shell):
        cluster = self.cluster(self.namespace("staging", "org/peer"), peers=1)
        run = subprocess.run

        def local_remote(command, **kwargs):
            return run([*shell, "-c", command[-1]], **kwargs)

        for state in ("missing", "empty", "populated"):
            with self.subTest(state=state):
                stage = self.root / state
                cluster.stage_dir = str(stage)
                if state != "missing":
                    stage.mkdir()
                stale, preserved = [], []
                if state == "populated":
                    stale = [stage / "serve_rank1.ops", stage / "serve_rank3.ops"]
                    (stage / "nested").mkdir()
                    preserved = [stage / "serve_r1.log", stage / "other.ops",
                                 stage / "nested/serve_rank1.ops"]
                    for path in stale + preserved:
                        path.write_text("previous run\n")
                with patch.object(subprocess, "run", side_effect=local_remote), \
                        patch.object(cluster, "scp_to", return_value=True), \
                        redirect_stdout(io.StringIO()) as output:
                    self.assertTrue(cluster.stage(), output.getvalue())
                self.assertTrue(stage.is_dir())
                for path in stale:
                    self.assertFalse(path.exists(), f"stale operation stream survived: {path}")
                for path in preserved:
                    self.assertEqual(path.read_text(), "previous run\n")

    def test_peer_staging_cleans_stale_streams_in_bash(self):
        self.check_peer_staging(["bash"])

    def test_peer_staging_cleans_stale_streams_with_failglob(self):
        # Exercise unmatched-glob errors even on hosts without zsh installed.
        self.check_peer_staging(["bash", "-O", "failglob"])

    @unittest.skipUnless(shutil.which("zsh"), "zsh is not installed")
    def test_peer_staging_cleans_stale_streams_in_zsh(self):
        self.check_peer_staging(["zsh", "-f"])

    def test_missing_peer_helper_and_invalid_status_are_not_reported_down(self):
        cluster = self.cluster(self.namespace("aaaa", "org/peer"), peers=1)
        stage = self.root / "peer-stage"
        stage.mkdir()
        cluster.stage_dir = str(stage)
        run = subprocess.run

        def local_remote(command, **kwargs):
            return run(["bash", "-c", command[-1]], **kwargs)

        with patch.object(subprocess, "run", side_effect=local_remote):
            self.assertIsNone(cluster.remote_control("status", 1, "peer1"))
            (stage / "rank1.process.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "control helper is missing"):
                cluster.remote_control("status", 1, "peer1")
            (stage / "cluster_process.py").write_text("print('{\"unexpected\": true}')\n")
            with self.assertRaisesRegex(RuntimeError, "invalid process status response"):
                cluster.remote_control("status", 1, "peer1")


if __name__ == "__main__":
    unittest.main()
