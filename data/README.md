# Organização dos dados

Cada dataset fica em uma pasta própria:

```text
data/<dataset>/
├── README.md
├── raw/                  # vídeos originais
├── metadata/             # manifesto e organização das amostras
└── processed/
    ├── landmarks/        # coordenadas extraídas
    ├── videos/           # vídeos transformados
    └── embeddings/       # representações dos modelos
```

- Coloque os vídeos diretamente em `raw/`, preservando nomes e conteúdo.
- Salve os resultados de processamento em `processed/`, separados por método e versão.
- Guarde a configuração usada e preserve a relação com o `sample_id` original.
- Versione `metadata/` e os READMEs. Não versione `raw/` nem `processed/`.
- Registre a origem e a licença no README de cada dataset.

## Manifesto

`metadata/manifest.csv` contém uma linha por vídeo, com identificação, rótulo,
sinalizador, repetição e metadados técnicos.

O campo `path` é relativo à raiz do dataset: `raw/arquivo.mp4`.
O `class_id` é o identificador original, não o índice de treinamento do PyTorch.

`labels.csv` e `splits.csv` serão adicionados na etapa de preparação dos experimentos.
