"""HTTP and lifecycle tests with no model downloads, API keys or live database."""
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app_main import create_app
from backend.core.config import PROJECT_ROOT, Settings, get_settings
from backend.core.runtime import Runtime, create_gemini_model, create_qdrant_client, load_runtime
from backend.rag.index_contract import identity


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, **kwargs):
        return SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])


class FakeQdrant:
    def __init__(self):
        self.error = None
        self.closed = False
        self.info = SimpleNamespace(
            points_count=1,
            config=SimpleNamespace(params=SimpleNamespace(
                vectors=SimpleNamespace(size=3, distance="Cosine"),
            )),
        )

    def get_collection(self, collection):
        if self.error:
            raise self.error
        return self.info

    def search(self, **kwargs):
        if self.error:
            raise self.error
        return [SimpleNamespace(score=0.9, payload={"title": "CDC", "page": 1, "content": "Trecho de teste"})]

    def retrieve(self, *args, **kwargs):
        return [SimpleNamespace(payload=identity(Settings(_env_file=None), 3))]

    def scroll(self, *args, **kwargs):
        records = [SimpleNamespace(payload={"doc_id": "test", "version": "v1"})] if self.info.points_count else []
        return records, None

    def close(self):
        self.closed = True


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = Settings(_env_file=None)
        self.qdrant = FakeQdrant()
        self.probe = FakeQdrant()
        self.runtime = Runtime(model=FakeModel(), qdrant=self.qdrant, probe=self.probe)
        self.loader = Mock(return_value=self.runtime)
        self.app = create_app(self.config, resource_loader=self.loader)

    def test_lifespan_loads_once_and_closes_clients(self):
        self.loader.assert_not_called()
        with TestClient(self.app) as client:
            self.loader.assert_called_once_with(self.config)
            for _ in range(2):
                self.assertEqual(client.get("/ready").status_code, 200)
                self.assertEqual(client.post("/search", json={"query": "teste"}).status_code, 200)
            self.loader.assert_called_once()
        self.assertTrue(self.qdrant.closed)
        self.assertTrue(self.probe.closed)
        self.assertIsNone(self.app.state.runtime)

    def test_routes_present_once_and_openapi_valid(self):
        routes = Counter((route.path, method) for route in self.app.routes for method in getattr(route, "methods", []))
        for path, method in [("/health", "GET"), ("/ready", "GET"), ("/search", "POST"),
                             ("/chat", "POST"), ("/chat/", "POST"), ("/classify/", "POST")]:
            self.assertEqual(routes[path, method], 1)
        with TestClient(self.app) as client:
            schema = client.get("/openapi.json").json()
            ids = [operation["operationId"] for path in schema["paths"].values() for operation in path.values()]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(client.get("/ui/").status_code, 200)
            self.assertEqual(client.get("/", follow_redirects=False).headers["location"], "/ui/")
            self.assertEqual(client.post("/classify/", json={"text": "prazo no CDC"}).status_code, 200)

    def test_no_gemini_key_is_required_for_search_and_chat_sources(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get("/ready").json()["checks"]["gemini"], "disabled")
            response = client.post("/chat", json={"question": "Quais são os direitos do consumidor?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sources"][0]["title"], "CDC")
            self.assertIn("não configurada", response.json()["answer"])

    def test_process_remains_alive_when_database_fails_and_recovers(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            self.probe.error = ConnectionError("secret-test-sentinel")
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["qdrant"], "unavailable")
            self.assertNotIn("secret-test-sentinel", response.text)
            self.assertEqual(client.get("/health").status_code, 200)
            self.probe.error = None
            self.assertEqual(client.get("/ready").status_code, 200)

    def test_empty_collection_is_not_ready(self):
        self.probe.info.points_count = 0
        with TestClient(self.app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["collection"], "empty")

    def test_missing_collection_is_not_ready(self):
        class MissingCollection(Exception):
            status_code = 404
        self.probe.error = MissingCollection()
        with TestClient(self.app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["collection"], "missing")
            self.assertEqual(response.json()["checks"]["qdrant"], "ready")

    def test_incompatible_vectors_are_not_ready(self):
        self.probe.info.config.params.vectors.size = 384
        with TestClient(self.app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["collection"], "incompatible")

    def test_probe_success_does_not_mask_missing_search_client(self):
        self.runtime.qdrant = None
        with TestClient(self.app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["qdrant"], "unavailable")

    def test_grpc_missing_collection_is_reported(self):
        class MissingCollection(Exception):
            def code(self):
                return SimpleNamespace(name="NOT_FOUND")
        self.probe.error = MissingCollection()
        with TestClient(self.app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["collection"], "missing")

    def test_unavailable_model_preserves_routes_and_reports_503(self):
        self.runtime.model = None
        with TestClient(self.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/ui/").status_code, 200)
            self.assertEqual(client.post("/classify/", json={"text": "prazo"}).status_code, 200)
            self.assertEqual(client.post("/search", json={"query": "teste"}).status_code, 503)
            self.assertEqual(client.post("/chat", json={"question": "Quais são os direitos do consumidor?"}).status_code, 503)

    def test_search_outage_is_503_without_exception_details(self):
        self.qdrant.error = ConnectionError("secret-test-sentinel")
        with TestClient(self.app) as client:
            response = client.post("/search", json={"query": "teste"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("secret-test-sentinel", response.text)

    def test_pending_lifespan_does_not_report_ready(self):
        client = TestClient(self.app)
        self.addCleanup(client.close)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/ready").status_code, 503)
        self.assertEqual(client.post("/chat", json={"question": "oi"}).status_code, 503)
        self.loader.assert_not_called()

    def test_cors_preflight_rejects_external_origins(self):
        with TestClient(self.app) as client:
            for origin, expected in [("http://localhost:8000", 200), ("https://evil.example", 400), ("null", 400)]:
                with self.subTest(origin=origin):
                    response = client.options("/chat", headers={
                        "Origin": origin, "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    })
                    self.assertEqual(response.status_code, expected)
                    self.assertNotIn("access-control-allow-credentials", response.headers)
                    self.assertEqual(response.headers.get("access-control-allow-origin"), origin if expected == 200 else None)

    def test_invalid_search_limits_are_rejected(self):
        with TestClient(self.app) as client:
            for limit in (-1, 0, 21):
                self.assertEqual(client.post("/search", json={"query": "teste", "top_k": limit}).status_code, 422)

    def test_resource_initialization_failure_is_visible_without_hiding_routes(self):
        with patch("backend.rag.embeddings.load_embedding_model", side_effect=OSError("download failed")), \
             patch("backend.core.runtime.create_qdrant_client", side_effect=[self.qdrant, self.probe]), \
             patch("backend.core.runtime.create_gemini_model") as gemini:
            runtime = load_runtime(self.config)
        gemini.assert_not_called()
        self.assertIsNone(runtime.model)
        self.assertFalse(runtime.readiness(self.config)[0])
        runtime.close()

    def test_gemini_misconfiguration_is_not_ready(self):
        self.config = Settings(_env_file=None, GEMINI_API_KEY="unused-test-value", GEMINI_MODEL="")
        app = create_app(self.config, resource_loader=self.loader)
        with TestClient(app) as client:
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.post("/chat", json={"question": "oi"}).status_code, 503)

    def test_ui_path_does_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                app = create_app(self.config, resource_loader=self.loader)
                with TestClient(app) as client:
                    self.assertEqual(client.get("/ui/").status_code, 200)
            finally:
                os.chdir(previous)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

    def test_defaults_and_relative_paths_are_independent_of_cwd(self):
        settings = Settings(_env_file=None, DATA_DIR="data")
        self.assertEqual(settings.DATA_DIR, PROJECT_ROOT / "data")
        self.assertEqual(Settings.model_config["env_file"], PROJECT_ROOT / ".env")
        self.assertFalse(settings.gemini_enabled)

    def test_environment_overrides_dotenv_and_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("QDRANT_HOST=file-host\nUNRELATED_SETTING=ignored\n", encoding="utf-8")
            with patch.dict(os.environ, {"QDRANT_HOST": "process-host"}):
                settings = Settings(_env_file=env_file)
            self.assertEqual(settings.QDRANT_HOST, "process-host")

    def test_example_can_be_loaded_without_credentials(self):
        settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
        self.assertFalse(settings.gemini_enabled)
        self.assertEqual(settings.EMBEDDING_DEVICE, "cpu")
        self.assertIsNone(settings.QDRANT_URL)

    def test_secrets_are_redacted_and_only_generation_requires_key(self):
        settings = Settings(_env_file=None, GEMINI_API_KEY="unused-test-value")
        self.assertNotIn("unused-test-value", repr(settings))
        with self.assertRaisesRegex(ValueError, "GEMINI_MODEL"):
            settings.require_gemini()
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            create_gemini_model(Settings(_env_file=None))

    def test_invalid_configuration_fails_early(self):
        for values in [{"QDRANT_PORT": 0}, {"CHUNK_SIZE": 100, "CHUNK_OVERLAP": 150}, {"READINESS_TIMEOUT": 0}]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(_env_file=None, **values)

    def test_probe_uses_short_timeout_and_url_takes_precedence(self):
        config = Settings(_env_file=None, QDRANT_URL="https://example.org", QDRANT_HOST="ignored")
        with patch("qdrant_client.QdrantClient") as factory:
            create_qdrant_client(config, probe=True)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["url"], "https://example.org")
        self.assertNotIn("host", kwargs)
        self.assertEqual(kwargs["timeout"], config.READINESS_TIMEOUT)
        self.assertEqual(kwargs["prefer_grpc"], config.QDRANT_USE_GRPC)
        self.assertIsNone(kwargs["port"])


class ImportTests(unittest.TestCase):
    def run_python(self, code):
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), GEMINI_API_KEY="", GEMINI_MODEL="")
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run([sys.executable, "-B", "-c", code], cwd=directory, env=env,
                                  capture_output=True, text=True, timeout=20)

    def test_import_does_not_load_models_or_gemini_sdk(self):
        result = self.run_python("import sys; import backend.app_main; assert 'sentence_transformers' not in sys.modules; assert 'google.generativeai' not in sys.modules")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_broken_router_import_is_fatal(self):
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'backend.api.routes_classify':
        raise ImportError('router-import-sentinel')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import backend.app_main
"""
        result = self.run_python(code)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("router-import-sentinel", result.stderr)


if __name__ == "__main__":
    unittest.main()
