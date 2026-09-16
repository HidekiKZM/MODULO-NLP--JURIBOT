"""The incomplete legacy pipeline is explicitly disabled.

ChatService is the sole chat flow, created by the application lifespan and used
by /chat. This module is retained to make outdated integrations fail clearly,
without loading a second model/client or silently changing response contracts.
"""


class RAGPipeline:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "RAGPipeline foi desativado. Use /chat ou o ChatService da aplicação. "
            "Streaming ainda não é suportado."
        )
