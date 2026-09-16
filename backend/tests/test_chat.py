"""Gemini responses and HTTP errors are simulated; no API credentials are used."""
import asyncio
import json
import io
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi.testclient import TestClient
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from backend.app_main import create_app
from backend.core.config import Settings
from backend.core.runtime import Runtime, SearchUnavailable
from backend.rag.generator_gemini import GeneratedAnswer, GeminiGenerator, validate_response
from backend.rag.pipeline import RAGPipeline
from backend.services.chat_errors import ChatFailure


def config(**overrides):
    return Settings(_env_file=None, GEMINI_API_KEY="unused-test-key", GEMINI_MODEL="gemini-3.8-flash",
                    GEMINI_RETRY_DELAY_SECONDS=0, **overrides)


def response(text="Resposta [1]", reason="STOP", **kwargs):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=reason, content=types.Content(role="model", parts=[types.Part(text=text)]),
    )], **kwargs)


def fake_client(*outcomes):
    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(side_effect=list(outcomes))),
                            aclose=AsyncMock()), close=Mock(),
    )


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "secret-provider-sentinel"}})


class GeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_smoke_sends_only_fixed_prompt_once_and_closes_client(self):
        from backend.scripts import smoke_gemini
        generator = SimpleNamespace(generate=AsyncMock(return_value=GeneratedAnswer("OK", {})), aclose=AsyncMock())
        output = io.StringIO()
        with patch.object(smoke_gemini, "GeminiGenerator", return_value=generator), redirect_stdout(output):
            self.assertEqual(await smoke_gemini.run_smoke(config()), 0)
        generator.generate.assert_awaited_once()
        self.assertEqual(generator.generate.call_args.args, ("Responda apenas com a palavra OK.",))
        generator.aclose.assert_awaited_once()
        self.assertTrue(json.loads(output.getvalue())["expected_answer"])

    async def test_success_limits_and_usage(self):
        client = fake_client(response(usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=4, total_token_count=16)))
        generator = GeminiGenerator(config(GEMINI_MAX_OUTPUT_TOKENS=123), client=client)
        answer = await generator.generate("pergunta", system_instruction="instruções")
        self.assertEqual(answer.text, "Resposta [1]")
        self.assertEqual(answer.usage["total_token_count"], 16)
        kwargs = client.aio.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(kwargs["config"].max_output_tokens, 123)
        self.assertEqual(kwargs["config"].candidate_count, 1)
        self.assertTrue(kwargs["config"].automatic_function_calling.disable)
        self.assertEqual(kwargs["config"].system_instruction, "instruções")
        await generator.aclose()
        client.aio.aclose.assert_awaited_once()
        client.close.assert_called_once()

    async def test_transient_failure_recovers_and_attempts_are_bounded(self):
        for error in [api_error(503), api_error(502), httpx.ConnectError("secret"),
                      httpx.ReadTimeout("secret"), asyncio.TimeoutError()]:
            with self.subTest(error=type(error).__name__):
                client = fake_client(error, response())
                generator = GeminiGenerator(config(), client=client)
                self.assertEqual((await generator.generate("q", system_instruction="s")).text, "Resposta [1]")
                self.assertEqual(client.aio.models.generate_content.await_count, 2)
        client = fake_client(api_error(503), api_error(503), response())
        generator = GeminiGenerator(config(), client=client)
        with self.assertRaises(ChatFailure) as raised:
            await generator.generate("q", system_instruction="s")
        self.assertEqual(raised.exception.code, "gemini_unavailable")
        self.assertEqual(client.aio.models.generate_content.await_count, 2)

    async def test_permanent_errors_never_retry_or_leak_provider_details(self):
        for status, expected in [(429, "gemini_quota_exceeded"), (401, "gemini_configuration"),
                                 (403, "gemini_configuration"), (404, "gemini_configuration"),
                                 (400, "gemini_configuration"), (418, "gemini_invalid_response")]:
            with self.subTest(status=status):
                client = fake_client(api_error(status), response())
                with self.assertRaises(ChatFailure) as raised:
                    await GeminiGenerator(config(), client=client).generate("q", system_instruction="s")
                self.assertEqual(raised.exception.code, expected)
                self.assertNotIn("secret", json.dumps(raised.exception.detail()))
                self.assertEqual(client.aio.models.generate_content.await_count, 1)

    async def test_total_deadline_cancels_pending_request(self):
        cancelled = asyncio.Event()
        async def pending(**kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
        client = fake_client()
        client.aio.models.generate_content.side_effect = pending
        with self.assertRaises(ChatFailure) as raised:
            await GeminiGenerator(config(GEMINI_TIMEOUT_SECONDS=0.03), client=client).generate("q", system_instruction="s")
        self.assertEqual(raised.exception.code, "gemini_timeout")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(client.aio.models.generate_content.await_count, 1)

    async def test_timeout_exception_is_distinct_after_last_attempt(self):
        client = fake_client(httpx.ReadTimeout("secret"))
        with self.assertRaises(ChatFailure) as raised:
            await GeminiGenerator(config(GEMINI_MAX_ATTEMPTS=1), client=client).generate("q", system_instruction="s")
        self.assertEqual(raised.exception.code, "gemini_timeout")

    async def test_invalid_blocked_and_truncated_results_never_retry(self):
        outcomes = [
            (response(reason="MAX_TOKENS"), "gemini_truncated"),
            (response(reason="SAFETY"), "gemini_blocked"),
            (types.GenerateContentResponse(prompt_feedback={"block_reason": "SAFETY"}), "gemini_blocked"),
            (response(text="  "), "gemini_invalid_response"),
            (types.GenerateContentResponse(), "gemini_invalid_response"),
            (response(reason="OTHER"), "gemini_invalid_response"),
        ]
        for outcome, expected in outcomes:
            with self.subTest(expected=expected):
                client = fake_client(outcome, response())
                with self.assertRaises(ChatFailure) as raised:
                    await GeminiGenerator(config(), client=client).generate("q", system_instruction="s")
                self.assertEqual(raised.exception.code, expected)
                self.assertEqual(client.aio.models.generate_content.await_count, 1)

    async def test_real_sdk_mock_transport_honors_retry_budget(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE", "message": "secret"}})
        real_client = genai.Client
        def factory(**kwargs):
            options = kwargs["http_options"]
            self.assertEqual(options.retry_options.attempts, 1)
            self.assertEqual(options.timeout, 30000)
            options.async_client_args = {"transport": httpx.MockTransport(handler)}
            return real_client(**kwargs)
        with patch("backend.rag.generator_gemini.genai.Client", side_effect=factory):
            generator = GeminiGenerator(config())
        try:
            with self.assertRaises(ChatFailure) as raised:
                await generator.generate("q", system_instruction="s")
            self.assertEqual(raised.exception.code, "gemini_unavailable")
            self.assertEqual(len(requests), 2, "SDK retries multiplied the configured budget")
            body = json.loads(requests[0].content)
            self.assertEqual(body["generationConfig"]["maxOutputTokens"], 1024)
        finally:
            await generator.aclose()

    async def test_cancellation_is_not_converted_into_a_generation_error(self):
        client = fake_client(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await GeminiGenerator(config(), client=client).generate("q", system_instruction="s")
        self.assertEqual(client.aio.models.generate_content.await_count, 1)


class ResponseValidationTests(unittest.TestCase):
    def test_real_smoke_requires_explicit_authorization_flag(self):
        from backend.scripts import smoke_gemini
        with patch("sys.argv", ["smoke_gemini"]), patch.object(smoke_gemini, "GeminiGenerator") as factory, \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            smoke_gemini.main()
        self.assertEqual(raised.exception.code, 2)
        factory.assert_not_called()

    def test_authorized_smoke_has_one_attempt_and_small_limits_without_calling_api(self):
        from backend.scripts import smoke_gemini
        with patch("sys.argv", ["smoke_gemini", "--authorize-api-call"]), \
             patch.object(smoke_gemini, "get_settings", return_value=config()), \
             patch.object(smoke_gemini, "run_smoke", new_callable=AsyncMock, return_value=0) as run:
            self.assertEqual(smoke_gemini.main(), 0)
        settings = run.call_args.args[0]
        self.assertEqual(settings.GEMINI_MAX_ATTEMPTS, 1)
        self.assertEqual(settings.GEMINI_MAX_OUTPUT_TOKENS, 256)
        self.assertEqual(settings.GEMINI_TIMEOUT_SECONDS, 15)

    def test_thoughts_are_not_returned(self):
        result = response()
        result.candidates[0].content.parts.insert(0, types.Part(text="private reasoning", thought=True))
        self.assertEqual(validate_response(result).text, "Resposta [1]")
        result.candidates[0].content.parts.pop()
        with self.assertRaises(ChatFailure):
            validate_response(result)

    def test_tool_calls_and_missing_finish_reason_are_not_success(self):
        result = response()
        result.candidates[0].content.parts = [types.Part(function_call=types.FunctionCall(name="unexpected"))]
        with self.assertRaises(ChatFailure):
            validate_response(result)
        with self.assertRaises(ChatFailure):
            validate_response(response(reason=None))

    def test_configuration_bounds(self):
        for values in [{"GEMINI_MAX_ATTEMPTS": 4}, {"GEMINI_TIMEOUT_SECONDS": 0},
                       {"GEMINI_TIMEOUT_SECONDS": float("inf")}, {"GEMINI_MAX_OUTPUT_TOKENS": 0}]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                config(**values)

    def test_legacy_pipeline_fails_explicitly_without_loading_resources(self):
        with self.assertRaisesRegex(RuntimeError, "desativado"):
            RAGPipeline()


class ChatHTTPTests(unittest.TestCase):
    def make_app(self, outcome, settings=None):
        settings = settings or config()
        sdk = fake_client(outcome, outcome)
        generator = GeminiGenerator(settings, client=sdk)
        runtime = Runtime(gemini=generator)
        runtime.search = Mock(return_value=[{"title": "CDC", "page": 1, "snippet": "Prazo de sete dias."}])
        return create_app(settings, resource_loader=lambda _: runtime), sdk, runtime

    def test_both_routes_share_service_preserve_citations_and_close_provider(self):
        app, sdk, runtime = self.make_app(response())
        with TestClient(app) as client:
            service = app.state.chat_service
            for path in ["/chat", "/chat/"]:
                result = client.post(path, json={"question": "Qual o prazo de arrependimento?"})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["answer"], "Resposta [1]")
                self.assertEqual(result.json()["sources"][0]["title"], "CDC")
                self.assertEqual(result.json()["generation"]["status"], "complete")
                self.assertIs(app.state.chat_service, service)
        sdk.aio.aclose.assert_awaited_once()
        sdk.close.assert_called_once()
        self.assertIsNone(app.state.chat_service)

    def test_provider_failures_have_distinct_non_success_http_codes(self):
        cases = [(api_error(429), 429, "gemini_quota_exceeded"),
                 (api_error(503), 503, "gemini_unavailable"),
                 (httpx.ReadTimeout("secret"), 504, "gemini_timeout"),
                 (response(reason="SAFETY"), 422, "gemini_blocked"),
                 (response(reason="MAX_TOKENS"), 502, "gemini_truncated"),
                 (response(text=""), 502, "gemini_invalid_response"),
                 (api_error(401), 503, "gemini_configuration")]
        for outcome, status, code in cases:
            with self.subTest(code=code):
                app, sdk, _ = self.make_app(outcome, config(GEMINI_MAX_ATTEMPTS=1))
                with TestClient(app) as client:
                    result = client.post("/chat", json={"question": "Oi"})
                    self.assertEqual(result.status_code, status)
                    self.assertEqual(result.json()["detail"]["code"], code)
                    self.assertNotIn("answer", result.json())
                    self.assertNotIn("secret", result.text)

    def test_search_failure_does_not_call_gemini(self):
        app, sdk, runtime = self.make_app(response())
        runtime.search.side_effect = SearchUnavailable("secret-db-error")
        with TestClient(app) as client:
            result = client.post("/chat", json={"question": "Qual o prazo de arrependimento?"})
            self.assertEqual(result.status_code, 503)
            self.assertEqual(result.json()["detail"]["code"], "search_unavailable")
            self.assertNotIn("secret", result.text)
        sdk.aio.models.generate_content.assert_not_called()

    def test_no_rag_path_does_not_search_and_blank_questions_are_rejected(self):
        app, sdk, runtime = self.make_app(response())
        with TestClient(app) as client:
            self.assertEqual(client.post("/chat", json={"question": "Oi"}).status_code, 200)
            self.assertEqual(client.post("/chat", json={"question": "   "}).status_code, 422)
        runtime.search.assert_not_called()
        self.assertEqual(sdk.aio.models.generate_content.await_count, 1)


if __name__ == "__main__":
    unittest.main()
