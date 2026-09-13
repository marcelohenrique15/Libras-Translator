# Organização dos dados

Cada dataset deve possuir seu próprio diretório dentro de `data/` e seguir a mesma estrutura:

```text
data/
├── README.md
└── <dataset>/
    ├── README.md
    ├── raw/
    │   └── videos/
    ├── metadata/
    │   ├── manifest.csv
    │   ├── labels.csv
    │   └── splits.csv
    └── processed/
        ├── landmarks/
        ├── videos/
        └── embeddings/
```

## `raw/`

Contém os arquivos originais do dataset. Vídeos devem ser colocados em:

```text
data/<dataset>/raw/
```

Esses arquivos não devem ser renomeados, editados ou sobrescritos. Todo processamento deve gerar novos arquivos em `processed/`.

## `metadata/`

Contém arquivos pequenos que descrevem e organizam o dataset:

- `manifest.csv`: uma linha por amostra, com caminho, classe, sinalizador e repetição;
- `labels.csv`: mapeamento entre identificadores, índices numéricos e nomes das classes;
- `splits.csv`: definição das amostras de treino, validação e teste.

Os metadados devem ser versionados pelo Git para que os experimentos possam ser reproduzidos.

## `processed/`

Contém somente dados derivados de `raw/`. Cada representação deve ser separada pelo tipo de processamento e pela versão do método. Exemplo:

```text
<dataset>/processed/
├── landmarks/
│   ├── samples/
│   ├── config.yaml
│   └── manifest.csv/
├── videos/
│   ├── samples/
│   ├── config.yaml
│   └── manifest.csv/
└── embeddings/
    ├── samples/
    ├── config.yaml
    └── manifest.csv/
```

- `landmarks/`: coordenadas extraídas por MediaPipe ou outro detector;
- `videos/`: recortes ou transformações de vídeo que precisem ser persistidos;
- `embeddings/`: representações pré-computadas por modelos como V-JEPA.

Cada processamento deve guardar sua configuração e, quando aplicável, um manifesto próprio.
Os artefatos devem usar o `sample_id` definido no manifesto original. Isso permite relacionar vídeo, landmarks e embeddings da mesma amostra.

## Regras

1. Nunca modificar arquivos em `raw/`.
2. Não versionar `raw/` nem `processed/` no Git.
3. Versionar os arquivos de `metadata/`.
4. Não usar nomes vagos como `final`, `final2` ou `novo` para processamentos.
5. Criar uma nova versão quando o método ou seus parâmetros mudarem de forma incompatível.
6. Documentar origem, licença e particularidades no `README.md` de cada dataset.
