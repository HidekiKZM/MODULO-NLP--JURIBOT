"""One explicitly authorized API call; synthetic prompt only, no RAG or PDFs."""
import argparse
import asyncio
import json

from backend.core.config import get_settings
from backend.rag.generator_gemini import GeminiGenerator
from backend.services.chat_errors import ChatFailure

PROMPT = "Responda apenas com a palavra OK."


async def run_smoke(settings):
    generator = GeminiGenerator(settings)
    try:
        result = await generator.generate(PROMPT, system_instruction="Teste técnico de conectividade. Seja breve.")
        matches = result.text.strip().rstrip(".! ").upper() == "OK"
        print(json.dumps({"model": settings.GEMINI_MODEL, "status": "complete",
                          "expected_answer": matches, "usage": result.usage}))
        return 0 if matches else 1
    finally:
        await generator.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-api-call", action="store_true", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    args = parser.parse_args()
    settings = get_settings().model_copy(update={
        "GEMINI_MODEL": args.model.strip(), "GEMINI_TIMEOUT_SECONDS": 15,
        "GEMINI_MAX_OUTPUT_TOKENS": 256, "GEMINI_MAX_ATTEMPTS": 1,
    })
    try:
        settings.require_gemini()
        return asyncio.run(run_smoke(settings))
    except ChatFailure as error:
        print(json.dumps({"status": "failed", **error.detail()}, ensure_ascii=False))
        return 1
    except Exception as error:
        # No raw exception/traceback: configuration errors can contain credentials.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "message": "Verifique a chave local e a configuração do SDK."}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
