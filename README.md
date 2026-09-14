# Libras Translator

Pesquisa para reconhecimento de sinais isolados de Libras.

## Instalação

Em Debian/Ubuntu com Python 3.12 disponível nos repositórios:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv ffmpeg
git clone https://github.com/marcelohenrique15/Libras-Translator.git
cd Libras-Translator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
```

O projeto exige Python 3.12. O `pyproject.toml` não instala o interpretador.

## Uso

Execute na raiz do repositório, com `.venv` ativada.

Coloque os vídeos em `data/<dataset>/raw/`. Veja a [estrutura dos dados](data/README.md).

Auditar uma pasta de vídeos:

```bash
libras-translator --audit data/minds_libras/raw
```

Para vídeos processados, passe a pasta que contém os arquivos.
A auditoria lê metadados, não percorre subpastas nem decodifica todos os frames.

Auditar e gerar o manifesto:

```bash
libras-translator --build-manifest data/minds_libras
```

Gera `data/minds_libras/metadata/manifest.csv` se não houver erros.

- `--overwrite`: substitui um manifesto existente.
- `--output caminho/manifest.csv`: muda o destino do CSV.
- `--help`: mostra todas as opções.
