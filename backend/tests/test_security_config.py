"""Security boundaries, checked without importing models or contacting services.

Run from the repository root:
    python -B -m unittest discover -s backend/tests -p test_security_config.py -v
"""
import ast
import json
from pathlib import Path
import subprocess
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]


class SecurityConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Never resolve .env contents or contact the Docker daemon.
        result = subprocess.run(
            ["docker", "compose", "-f", "ops/docker-compose.yml", "config",
             "--no-env-resolution", "--no-interpolate", "--format", "json"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        cls.compose = json.loads(result.stdout)

    def test_databases_have_no_host_ports(self):
        for name in ("qdrant", "redis"):
            with self.subTest(service=name):
                service = self.compose["services"][name]
                self.assertFalse(service.get("ports"))
                self.assertNotIn(service.get("network_mode"), ("host",))
                self.assertIn("juribot-net", service["networks"])

    def test_every_published_port_is_loopback_only(self):
        for name, service in self.compose["services"].items():
            for port in service.get("ports", []):
                with self.subTest(service=name, port=port["published"]):
                    self.assertEqual(port["host_ip"], "127.0.0.1")
        self.assertTrue(self.compose["services"]["api"]["ports"])

    def test_nextjs_is_optional(self):
        self.assertTrue(self.compose["services"]["frontend"].get("profiles"))
        self.assertNotIn("frontend", self.compose["services"]["api"]["depends_on"])

    def test_persistent_database_mounts_are_preserved(self):
        for name, source, target in (
            ("qdrant", "qdrant_data", "/qdrant/storage"),
            ("redis", "redis_data", "/data"),
        ):
            with self.subTest(service=name):
                self.assertTrue(any(
                    mount.get("type") == "volume"
                    and mount.get("source") == source
                    and mount.get("target") == target
                    for mount in self.compose["services"][name]["volumes"]
                ))
                self.assertIn(source, self.compose["volumes"])

    def test_cors_accepts_only_explicit_local_origins(self):
        tree = ast.parse((ROOT / "backend/app_main.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and node.args
                 and isinstance(node.args[0], ast.Name)
                 and node.args[0].id == "CORSMiddleware"]
        self.assertEqual(len(calls), 1)
        config = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
        self.assertFalse(config["allow_credentials"])
        self.assertTrue(config["allow_origins"])
        for origin in config["allow_origins"]:
            with self.subTest(origin=origin):
                url = urlsplit(origin)
                self.assertEqual(url.scheme, "http")
                self.assertIn(url.hostname, ("localhost", "127.0.0.1"))
                self.assertIn(url.port, (8000, 3000))
                self.assertFalse(url.username or url.password or url.path or url.query or url.fragment)
        self.assertNotIn("*", config["allow_methods"])
        self.assertNotIn("*", config["allow_headers"])

    def test_git_ignores_secrets_but_allows_examples_and_dockerignore(self):
        ignored = [".env", ".env.local", "backend/.env", "frontend/.env.production", "backend/private.env"]
        allowed = [".env.example", "backend/.env.example", "frontend/.env.example",
                   "backend/.dockerignore", "frontend/.dockerignore"]
        for path in ignored + allowed:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT,
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0 if path in ignored else 1)


if __name__ == "__main__":
    unittest.main()
