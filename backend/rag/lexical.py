# Arquivo: backend/rag/lexical.py

from rank_bm25 import BM25Okapi
from backend.core.config import get_settings

class LexicalSearch:
    """
    Gerencia a busca lexical (por palavras-chave).
    
    Atualmente implementado como um fallback em memória usando rank_bm25.
    A integração com OpenSearch seria uma expansão desta classe.
    """
    def __init__(self):
        self.use_opensearch = get_settings().USE_OPENSEARCH
        self.in_memory_index = None
        self.documents = []

    def index(self, documents: list[dict]):
        """
        Cria um índice BM25 em memória a partir dos documentos.
        
        Args:
            documents (list[dict]): Lista de documentos, onde cada um tem uma chave 'content'.
        """
        if self.use_opensearch:
            print("INFO: Busca lexical com OpenSearch ativada (implementação pendente).")
            # Aqui entraria a lógica para indexar no OpenSearch
            return

        print("INFO: Criando índice lexical BM25 em memória...")
        self.documents = documents
        tokenized_corpus = [doc["content"].split() for doc in documents]
        self.in_memory_index = BM25Okapi(tokenized_corpus)
        print("INFO: Índice lexical em memória criado.")

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Realiza uma busca por palavras-chave.
        
        Args:
            query (str): A string de busca do usuário.
            limit (int): O número de resultados a retornar.
            
        Returns:
            list[dict]: Uma lista de documentos relevantes.
        """
        if self.use_opensearch:
            # Lógica para buscar no OpenSearch
            return []

        if not self.in_memory_index:
            return []

        tokenized_query = query.split()
        # O BM25 retorna os documentos mais relevantes com base na frequência dos termos.
        top_docs = self.in_memory_index.get_top_n(tokenized_query, self.documents, n=limit)
        return top_docs
