## Requisitos

O ambiente de desenvolvimento requer:

- Git;
- Python 3.12;
- aproximadamente 100 GB disponíveis para dados e artefatos iniciais.

O `pyproject.toml` exige Python 3.12, mas não instala o interpretador automaticamente.

## Instalação no Ubuntu 24.04 e distribuições compatíveis

Atualize o índice de pacotes:

```bash
sudo apt update
```

Instale Git, Python 3.12 e o suporte a ambientes virtuais:

```bash
sudo apt install git python3.12 python3.12-venv
```

Confirme a instalação:

```bash
python3.12 --version
```

A saída deve começar com:

```text
Python 3.12
```

> Não substitua o Python padrão do sistema nem altere o comando global `python3`.
> O projeto usará Python 3.12 somente dentro de seu ambiente virtual.

## Configuração do ambiente de desenvolvimento

Clone o repositório:

```bash
git clone https://github.com/marcelohenrique15/Libras-Translator.git
cd Libras-Translator
```

Crie o ambiente virtual:

```bash
python3.12 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Confirme a versão utilizada:

```bash
python --version
```

Instale o projeto em modo editável:

```bash
python -m pip install --editable .
```


## Dataset

Os arquivos do dataset não são armazenados no Git.

Coloque o arquivo `data.zip` em:

```text
data/raw/data.zip
```

Não extraia o ZIP manualmente. O pipeline será responsável por inspecionar, validar e processar seu conteúdo de forma reproduzível.