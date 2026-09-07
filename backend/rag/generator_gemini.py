from backend.core.config import get_settings
from backend.core.runtime import create_gemini_model


class GeminiGenerator:
    """
    Esta classe encapsula toda a lógica para se comunicar com a API do Gemini.
    Ela é responsável por construir os prompts e gerar as respostas.
    """

    def __init__(self):
        """
        O construtor da classe. É executado sempre que um novo objeto GeminiGenerator é criado.
        Sua principal função é inicializar o modelo generativo que será usado.
        """
        # Carrega o modelo generativo especificado nas configurações (ex: 'gemini-1.5-flash').
        # O objeto 'self.model' será usado para fazer as chamadas à API.
        self.model = create_gemini_model(get_settings())

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Método privado para construir o prompt final que será enviado ao modelo.
        Esta é a parte mais importante da "Engenharia de Prompt", onde definimos
        como o modelo deve se comportar. A qualidade da resposta depende diretamente
        da clareza e da precisão deste prompt.

        Args:
            question (str): A pergunta feita pelo usuário.
            context (str): O trecho de texto relevante encontrado nos documentos (resultado do RAG).

        Returns:
            str: O prompt completo, formatado e pronto para ser enviado à API.
        """
        # Verifica se foi fornecido um contexto. Isso diferencia uma pergunta de RAG
        # de uma conversa geral.
        if context:
            # Se há contexto, instruímos o modelo a atuar como um especialista
            # que responde *apenas* com base nas informações fornecidas.
            system_prompt = f"""
            Você é o Juribot, um assistente jurídico especializado em legislação brasileira.
            Sua função é responder perguntas baseando-se ESTREITAMENTE no CONTEXTO fornecido.
            NÃO use nenhum conhecimento externo.

            REGRAS:
            1. Responda de forma clara e objetiva.
            2. Se a resposta estiver no contexto, responda e, ao final de cada parágrafo relevante,
               cite a fonte usando o formato [Fonte: Título do Documento, Página X].
            3. Se a informação não estiver no contexto, responda EXATAMENTE:
               "Com base nos documentos fornecidos, não encontrei informações suficientes para responder a sua pergunta."
            4. NÃO ofereça aconselhamento jurídico. Inicie toda resposta com o aviso:
               "Aviso: Esta é uma resposta gerada por IA e não substitui o aconselhamento de um profissional jurídico."

            CONTEXTO FORNECIDO:
            ---
            {context}
            ---
            """
        else:
            # Se não há contexto, o modelo é instruído a agir como um assistente
            # conversacional, recusando-se a responder perguntas jurídicas.
            system_prompt = """
            Você é o Juribot, um assistente conversacional.
            Responda de forma educada e prestativa a saudações ou perguntas gerais.
            NÃO responda a perguntas de natureza jurídica, pois você não tem acesso a documentos.
            Se o usuário fizer uma pergunta jurídica, peça para ele ser mais específico ou fornecer um documento.
            """

        # Concatena as instruções do sistema com a pergunta do usuário para formar o prompt final.
        return f"{system_prompt}\n\nPERGUNTA DO USUÁRIO: {question}\n\nRESPOSTA:"

    def generate_response(self, question: str, context: str = "", stream: bool = False):
        """
        Gera uma resposta do modelo com base na pergunta e no contexto.

        Args:
            question (str): A pergunta do usuário.
            context (str, optional): O contexto recuperado dos documentos. Padrão é "".
            stream (bool, optional): Define se a resposta será retornada de uma vez (False)
                                     ou em pedaços (streaming, True). Padrão é False.

        Returns:
            str or generator: Se stream=False, retorna a string da resposta completa.
                              Se stream=True, retorna um gerador que pode ser iterado
                              para obter os pedaços (chunks) da resposta.
        """
        # 1. Constrói o prompt completo usando o método auxiliar.
        full_prompt = self._build_prompt(question, context)

        # 2. Verifica se a resposta deve ser em modo streaming.
        if stream:
            # O modo streaming é ideal para interfaces de chat, pois a resposta aparece
            # palavra por palavra, melhorando a experiência do usuário.
            # A chamada `generate_content` com `stream=True` retorna um objeto iterável.
            response_stream = self.model.generate_content(full_prompt, stream=True)
            # Retornamos um "generator expression" que extrai o texto de cada pedaço (chunk).
            return (chunk.text for chunk in response_stream)
        else:
            # O modo padrão (não-streaming) espera a resposta completa do modelo
            # antes de retorná-la. É mais simples, mas pode parecer lento para o usuário.
            response = self.model.generate_content(full_prompt)
            # Retorna o texto extraído do objeto de resposta.
            return response.text
