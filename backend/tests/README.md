# Validação de segurança — etapa 1

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
