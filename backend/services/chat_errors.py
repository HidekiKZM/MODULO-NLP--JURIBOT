"""Public error contract. Never include provider messages, prompts or credentials."""

ERRORS = {
    "chat_initializing": (503, "Inicialização do chat pendente.", True),
    "search_unavailable": (503, "Busca indisponível; consulte /ready.", True),
    "gemini_unavailable": (503, "Serviço de geração temporariamente indisponível.", True),
    "gemini_configuration": (503, "Geração indisponível; verifique a configuração do Gemini.", False),
    "gemini_quota_exceeded": (429, "Quota ou limite de requisições do Gemini excedido.", False),
    "gemini_blocked": (422, "A geração foi bloqueada pelo provedor.", False),
    "gemini_truncated": (502, "A resposta atingiu o limite de saída e foi descartada.", False),
    "gemini_timeout": (504, "O tempo limite de geração foi excedido.", True),
    "gemini_invalid_response": (502, "O provedor retornou uma resposta inválida ou vazia.", False),
}


class ChatFailure(Exception):
    def __init__(self, code):
        self.code = code
        self.status, self.message, self.retryable = ERRORS[code]
        super().__init__(self.message)

    def detail(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}
