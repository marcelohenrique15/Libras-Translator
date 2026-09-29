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

- Coloque os vídeos em `raw/`, inclusive em subpastas por sinalizador, preservando os nomes.
- Salve os resultados de processamento em `processed/`, separados por método e versão.
- Guarde a configuração usada e preserve a relação com o `sample_id` original.
- Versione `metadata/` e os READMEs. Não versione `raw/` nem `processed/`.
- Registre a origem e a licença no README de cada dataset.

## Manifesto

`metadata/manifest.csv` contém uma linha por vídeo selecionado, com `sample_id`,
`class_id`, `label`, `signer_id`, `repetition`, `path` e `set` (`train` ou `test`).

O campo `path` é relativo à raiz do dataset, por exemplo `raw/01AcontecerSinalizador01-1.mp4`.
O `class_id` é o identificador original, não o índice de treinamento do PyTorch.
O manifesto é sobrescrito ao executar novamente `--build-manifest` com outro
`--test-signer-id`.
