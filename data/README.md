# Organização dos dados

```text
data/<dataset>/
├── README.md
├── raw/                              # vídeos originais
├── metadata/manifest.csv             # uma linha por amostra
└── processed/
    ├── landmarks/                    # coordenadas extraídas (.csv)
    ├── selected_landmarks/<subset>/<hash>/
    ├── imputed_landmarks/<subset>/<hash>/
    └── encoded_inputs/<representation>/<hash>/
```

As etapas seguintes usam `.npy`. Os hashes identificam código e opções que afetam a etapa. Reexecutar com a mesma configuração reutiliza o cache atual. Augmentation do treino não é salva nesse cache.

## Formatos de entrada

Para datasets gerais, organize os vídeos por classe e sinalizador:

```text
raw/<class_id>/<signer_id>/video.mp4
```

Subpastas dentro do sinalizador também são aceitas. `class_id` e `label` recebem o nome da pasta da classe; `signer_id` vem da pasta seguinte. `sample_id` é formado pelo caminho sem extensão, juntando segmentos com `__`.

```bash
libras-translator --build-manifest --dataset data/meu_dataset --dataset-format class_signer
```

`--dataset-format auto` reconhece os nomes MINDS-Libras ou usa a estrutura acima. Para explicitar MINDS, use `minds_libras`.

Também é possível fornecer seu próprio `metadata/manifest.csv`. Colunas mínimas:

```csv
sample_id,class_id,label,signer_id,path
amostra_01,ola,Olá,pessoa_a,raw/ola/pessoa_a/video.mp4
```

Use IDs como strings, um `sample_id` único e caminhos relativos à raiz do dataset. `repetition` é opcional num manifesto próprio. O builder a cria com o nome do vídeo no formato geral. O filtro de repetições 1 a 5 aplica-se somente ao builder MINDS-Libras.

## Divisão e resultados

```bash
libras-translator --split-manifest --dataset data/meu_dataset --test-signer-id pessoa_a
```

Acrescenta ou atualiza `set` (`train` ou `test`) sem descartar linhas nem colunas adicionais. A validação é definida durante o treino, por outro sinalizador. Com teste `all`, cada sinalizador é reservado em uma rodada externa distinta.

`runs/<dataset>/<run-id>/config.json` registra a configuração geral. Os arquivos de cada rodada ficam em `test_<ID>/val_<ID>/`:

- `history.json`: métricas por época.
- `best.pt`, `last.pt`: melhor checkpoint e ponto de retomada.
- `validation.json`: resultado interno usado na seleção.
- `test.json`, `result.json`: avaliação externa e resumo da rodada.

A busca Optuna salva `optuna.sqlite3`, `search_test<ID>.json` e `best_config_test<ID>.json` em `runs/<dataset>/search_<hash>/`. Cada sinalizador de teste tem um estudo próprio; as tentativas são escolhidas pelo F1-macro das validações. O banco guarda o histórico para continuar a busca até o orçamento de tentativas concluídas. Execute uma busca por processo e mantenha o banco em armazenamento local, sem NFS.

Pesos ImageNet ficam em `weights/checkpoints/`.

Vídeos, metadados, dados processados, resultados e pesos não são versionados. Versione o README de cada dataset com origem e licença.
