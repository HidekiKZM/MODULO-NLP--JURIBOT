# Juribot

Assistente jurídico experimental com FastAPI, interface HTML, busca semântica em PDFs no Qdrant e geração opcional pelo Gemini. O escopo atual de estabilização é a API e `backend/ui/index.html`. O Next.js permanece em `frontend/`, fora da inicialização padrão.

## Execução com Docker

Requisitos: Docker Engine/Desktop em execução e Docker Compose 2.24 ou superior. Execute na raiz do repositório:

```sh
docker compose -f ops/docker-compose.yml config --quiet
docker compose -f ops/docker-compose.yml build api
docker compose -f ops/docker-compose.yml up -d
```

Abra <http://127.0.0.1:8000/ui/>. Documentação da API: <http://127.0.0.1:8000/docs>. A API publica apenas em loopback; Qdrant e Redis ficam na rede do Compose.

O arquivo `.env` é opcional. Para personalizar, copie `.env.example` para `.env` e ajuste os valores localmente. Nunca versione credenciais. Sem `GEMINI_API_KEY`, busca, classificação e interface continuam disponíveis; o chat retorna fontes sem geração. Para gerar respostas, configure também `GEMINI_MODEL` com um modelo disponível na sua conta. Não há chamada paga no teste de prontidão.

O primeiro início baixa o modelo de embeddings e pode demorar; exige acesso à internet e espaço para o cache. A imagem padrão usa PyTorch para CPU. Modelos são carregados no ciclo de vida da API, sem downloads ao importar módulos. Uma falha de carregamento deixa a aplicação sem prontidão; após resolver a causa, reinicie a API.

```sh
docker compose -f ops/docker-compose.yml logs -f api
docker compose -f ops/docker-compose.yml restart api
```

## Dados e prontidão

- `GET /health`: processo respondendo, independentemente do banco ou dos modelos.
- `GET /ready`: 200 quando embeddings, Qdrant e coleção estão disponíveis e compatíveis; 503 enquanto houver dependência indisponível, coleção ausente/vazia ou configuração Gemini incompleta.
- `POST /search`, `POST /chat` e `POST /classify`: consulte os contratos em `/docs`.

Uma instalação sem índice retorna 503 em `/ready`. Para indexar PDFs existentes em `data/`, execute explicitamente:

```sh
docker compose -f ops/docker-compose.yml exec api python -m backend.ingest.prepare_index
```

A ingestão escreve no Qdrant. Use o mesmo `EMBEDDING_MODEL` da API; dimensões diferentes exigem planejar outra coleção/reindexação. A prontidão verifica dimensão e distância, mas não identifica modelos diferentes que produzam a mesma dimensão. O processo de ingestão não faz parte da inicialização automática.

PDFs são montados em `/app/data`; os volumes `qdrant_data`, `redis_data` e `embedding_cache` persistem. Não use `down --volumes` para interromper o projeto. Antes de iniciar sobre uma instalação existente, mantenha o nome de projeto Compose usado anteriormente: ele determina quais volumes serão associados. O padrão agora é `juribot`; use `-p NOME_EXISTENTE` em todos os comandos se necessário, sem apagar volumes.

## Execução Python local

Ambiente validado: Python 3.11. Crie um ambiente virtual e instale:

```sh
python -m venv .venv
# Ative .venv conforme seu sistema operacional.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app_main:app --host 127.0.0.1 --port 8000
```

Instale Poppler e Tesseract com idioma português para OCR local. No Windows, `POPPLER_PATH` pode apontar para os binários. Os caminhos relativos de `DATA_DIR` são resolvidos a partir da raiz do projeto, e a configuração lê o `.env` dessa raiz.

A API local precisa de Qdrant acessível no endereço configurado. O Compose padrão não publica o banco no host; prefira executar a API pelo Compose ou configurar um banco local separado. Dentro do Compose, `QDRANT_HOST=qdrant` é definido automaticamente. Um `QDRANT_URL` explícito tem precedência.

## Estrutura

| Caminho | Responsabilidade |
| --- | --- |
| `backend/app_main.py` | Aplicação, ciclo de vida, rotas e saúde |
| `backend/core/config.py` | Configuração única e validação |
| `backend/core/runtime.py` | Recursos, busca e prontidão |
| `backend/api/` | Chat e classificação |
| `backend/rag/` | Embeddings e componentes de recuperação |
| `backend/ingest/` | Extração, fragmentação e indexação |
| `backend/ui/` | Interface HTML principal |
| `backend/tests/` | Testes de inicialização e segurança |
| `frontend/` | Next.js opcional, ainda fora do escopo estabilizado |
| `ops/docker-compose.yml` | Serviços e volumes |
| `data/` | Documentos locais |
| `.github/workflows/ci.yml` | Testes e build automatizados |

## Testes e CI

```sh
python -m pip install -r backend/requirements-test.txt
python -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

Os testes de configuração também exigem Git e Docker Compose recente com `--no-env-resolution`. Consulte [os testes da interface](backend/tests/README.md) para executar as verificações DOM. A CI executa testes Python, segurança da interface, validação Compose, build, `pip check` e imports sem rede ou credenciais.

Com GNU Make instalado, `make config`, `make build`, `make up`, `make logs`, `make ingest` e `make test` usam os caminhos corretos. Para uma instalação anterior com outro nome de projeto, passe `COMPOSE_PROJECT_NAME=NOME_EXISTENTE`.

O perfil `--profile frontend` permite iniciar o Next.js explicitamente; seus arquivos foram preservados, mas esse fluxo ainda não foi estabilizado. Redis permanece na composição, embora não seja uma dependência da busca atual. O SDK Gemini legado e a qualidade/validação factual das respostas ainda precisam de revisão própria.
