# Libras Translator

Projeto de pesquisa para reconhecimento de sinais isolados de Libras.

## Preparar o ambiente

Em uma distribuição baseada em Debian, instale os requisitos:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv ffmpeg
```

Clone o repositório:

```bash
git clone https://github.com/marcelohenrique15/Libras-Translator.git
cd Libras-Translator
```

Crie e ative o ambiente virtual:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Instale o projeto:

```bash
python -m pip install --editable .
```

## Dataset

Os dados não são versionados pelo Git. Coloque os vídeos extraídos em:

```text
data/<nome_dataset>/raw/
```
Para mais informações de armazenamento dos dados, acesse `data/README.md`
