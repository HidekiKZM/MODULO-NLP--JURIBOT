# Testes de inicialização e segurança

## API e ciclo de vida

`test_ingestion.py` usa Qdrant em memória e embeddings determinísticos para testar
compatibilidade, valores monetários, lotes, versões, interrupções e repetição sem
duplicações. Não acessa as coleções persistentes nem baixa modelos.

O teste manual `python -m backend.tests.smoke_ingestion` requer as dependências
completas e um Qdrant descartável configurado por `QDRANT_HOST`/`QDRANT_URL`.
Ele baixa a revisão fixada do modelo, gera um PDF temporário e cria uma coleção
`juribot_validation_<UUID>` para validar busca, repetição e atualização reais.
Não exclui a coleção criada; use um servidor de teste sem volumes de produção.

Use Python 3.11 e execute na raiz:

```sh
python -m pip install -r backend/requirements-test.txt
python -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

Os testes HTTP usam recursos simulados, sem baixar modelos, acessar Gemini ou
alterar o Qdrant. Cobrem rotas, imports, configuração, CORS, fechamento dos
clientes e prontidão com falhas de dependências. A CI também constrói a imagem
completa e verifica suas dependências e imports sem rede.

Para executar os nove casos da interface em jsdom, instale `jsdom@26.1.0` em
uma pasta temporária, defina `NODE_PATH` para seu `node_modules` e execute
`node backend/tests/run_ui_security.cjs`. Esse teste automatizado não substitui
a conferência visual em navegador descrita abaixo.

## Segurança sem dependências Python da API

Execute na raiz do repositório. Não é necessário instalar as dependências da API,
criar `.env`, baixar modelos nem iniciar containers.

## Configuração

Requer Python 3.10+, Git e Docker Compose com `--no-env-resolution`.

```sh
python -B -m unittest discover -s backend/tests -p test_security_config.py -v
```

Verifica as portas publicadas, os volumes persistentes, o perfil opcional do
Next.js, a lista de origens CORS e as regras de exclusão do Git. A verificação de
CORS inspeciona a configuração sem importar a aplicação ou seus modelos; não
substitui um teste HTTP do middleware. O Compose é analisado sem ler segredos e
sem acessar o daemon Docker.

## Interface em navegador

```sh
python -B backend/tests/serve_ui_security.py
```

Abra <http://127.0.0.1:8765/tests/ui_security.html>. O resultado esperado é
`9 passaram; 0 falharam.` Encerre o servidor com Ctrl+C.

O servidor de teste disponibiliza apenas a página de testes e o HTML da interface,
em loopback. Os testes usam o DOM real do navegador, removem o script externo de
estilos da cópia de teste e simulam a resposta de `/chat`, sem enviar consultas ao
Gemini. Cobrem HTML malicioso, citações, links perigosos, atributos, dados
malformados e substituição de resultados anteriores.

As verificações não iniciam nem reconfiguram serviços já existentes. Aplicar as
novas portas a containers em execução exige recriação posterior dos serviços,
preservando os volumes.
