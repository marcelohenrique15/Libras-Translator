# Libras Translator

Pesquisa para reconhecimento de sinais isolados de Libras.

## Instalação

Com Python 3.12 instalado:

```bash
git clone https://github.com/marcelohenrique15/Libras-Translator.git
cd Libras-Translator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

O `pyproject.toml` instala MediaPipe e PyTorch, mas não o interpretador Python.

## Uso

Execute na raiz do repositório, com `.venv` ativada.

Coloque os vídeos em `data/<dataset>/raw/`. Veja a [estrutura dos dados](data/README.md).

Gere o manifesto e escolha o sinalizador de teste:

```bash
libras-translator --build-manifest data/minds_libras --test-signer-id 05
```

O comando cria ou sobrescreve `data/minds_libras/metadata/manifest.csv`.
A coluna `set` marca o sinalizador 05 como `test` e os demais como `train`.
