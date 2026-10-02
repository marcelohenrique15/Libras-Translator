# Libras Translator

Pesquisa para reconhecimento de sinais isolados de Libras, com modelos e processamento configuráveis.

## Instalação

Com Python 3.12, na raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Treino

```bash
libras-translator --train --dataset minds_libras --model resnet18
libras-translator --train --config configs/resnet18.toml
libras-translator --train --config configs/landmark_lstm.toml --epochs 50
libras-translator --list-models
```

`--dataset` aceita uma pasta em `data/` ou um caminho. A forma anterior `--train data/minds_libras` também funciona. Argumentos CLI substituem os valores do arquivo; opções específicas ficam nas tabelas `model_options` e `processor_options`. Também é possível carregar um `best_config_test<ID>.json` produzido pela busca.

O pipeline mostra os passos: manifesto → landmarks → seleção → imputação → codificação → treino → avaliação. Reutiliza arquivos atuais e rodadas concluídas. Uma rodada interrompida retoma da próxima época salva. Augmentation acontece somente no treino, antes da codificação.

| Opções | Uso |
|---|---|
| `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--patience` | Treino |
| `--test-signer-id ID\|all`, `--validation-signer-id ID` | Divisão por sinalizador |
| `--subset asl_2nd\|all\|arcanjo`, `--frame-count` | Pontos e tamanho das sequências |
| `--imputation` / `--no-imputation`, `--augmentation` / `--no-augmentation` | Processamento |
| `--processor NOME` | Processor registrado ou `pacote.modulo:Classe` |
| `--device auto\|cpu\|cuda:N`, `--num-workers`, `--seed` | Execução |
| `--output-dir`, `--weights-dir` | Resultados e pesos compartilhados |
| `--search-config`, `--search-trials` | Espaço de busca Optuna e total de tentativas concluídas |

Sem escolher o teste, usa o sinalizador marcado no manifesto ou o primeiro ID disponível. Cada outro sinalizador é usado para validação. `--validation-signer-id` restringe a uma rodada. Para LOPO completo:

```bash
libras-translator --train --config configs/resnet18.toml --test-signer-id all
```

Com 12 sinalizadores, são 132 rodadas. Treino, validação e teste usam sinalizadores diferentes. O melhor checkpoint e o early stopping seguem o **F1-macro de validação**. A loss usada nos gradientes continua sendo cross-entropy.

## Preparação por etapa

```bash
libras-translator --build-manifest --dataset minds_libras
libras-translator --split-manifest --dataset minds_libras --test-signer-id 05
libras-translator --extract-landmarks --dataset minds_libras
libras-translator --select-landmarks --config configs/resnet18.toml
libras-translator --impute-landmarks --config configs/resnet18.toml
libras-translator --encode-landmarks --config configs/resnet18.toml
```

`--build-manifest` recria o CSV; `--split-manifest` atualiza `set`, preservando as outras colunas. Etapas processadas usam cache por código e configuração. Arquivos incompletos são refeitos. Veja o [formato dos dados](data/README.md).

## Busca de hiperparâmetros

```bash
libras-translator --train --config configs/resnet18.toml --search-config configs/search.toml
```

O Optuna usa TPE para sugerir hiperparâmetros e **maximizar a média do F1-macro nas validações internas**. O teste externo avalia somente a configuração vencedora. Com `all`, cada sinalizador de teste tem um estudo próprio. O melhor resultado fica salvo; uma tentativa nova pode explorar uma configuração pior.

O exemplo tem `n_trials = 20`: vinte tentativas concluídas por estudo. O histórico fica em `optuna.sqlite3` no diretório da busca. Repetir o comando retoma o estudo e os checkpoints existentes. Para ampliar o orçamento para quarenta tentativas no total:

```bash
libras-translator --train --config configs/resnet18.toml \
  --search-config configs/search.toml --search-trials 40
```

Defina listas de opções ou intervalos no [arquivo de busca](configs/search.toml). Parâmetros de modelos e processors também podem ser ajustados; veja o [guia de extensão](docs/EXTENDING.md#configuração-e-busca).

## Organização e extensão

```text
src/cli/              # argumentos e chamada das etapas
src/dataset/          # manifesto, divisão e carregamento
src/preprocessor/     # extração, seleção, imputação e codificação
src/models/           # arquiteturas e registry
src/training/         # configuração, treino, busca e pipeline
configs/              # presets e espaços de busca
data/<dataset>/       # vídeos, manifesto e entradas processadas
runs/<dataset>/       # configurações, checkpoints e métricas
weights/checkpoints/  # pesos ImageNet compartilhados
tests/                # verificações do pipeline
```

Para adicionar modelos ou processors sem editar o pipeline, veja [como estender o projeto](docs/EXTENDING.md).

## Cluster

Prepare os dados e os pesos em um ambiente com acesso à rede antes do job. Os pesos ImageNet são baixados no primeiro uso; também podem ser copiados para `weights/checkpoints/`. Para sequências, prepare usando o preset da LSTM.

Para treinar sem transferir os vídeos, copie `metadata/manifest.csv`, os CSVs de landmarks e `processed/landmarks/index.json`. Preserve as datas dos CSVs; o índice confirma a extração completa sem os vídeos originais.

```bash
libras-translator --encode-landmarks --config configs/resnet18.toml
libras-translator --train --config configs/resnet18.toml \
  --test-signer-id 05 --validation-signer-id 01 \
  --device cuda:0 --num-workers 4 \
  --output-dir /scratch/libras/runs --weights-dir /scratch/libras/weights
```

Cada processo deve escrever em uma combinação exclusiva de teste e validação. O projeto executa um processo por GPU; não distribui um treino entre GPUs nem envia jobs ao Slurm. `cuda:0` considera as GPUs visíveis para o processo.

A busca Optuna executa tentativas em sequência. Use um processo por busca e mantenha `--output-dir` em armazenamento local do nó para o SQLite. Não compartilhe o mesmo banco entre processos nem use SQLite em NFS.

## Experimento de referência

O preset ResNet usa ASL-2nd com 80 pontos e imputação, baseado no [artigo](https://arxiv.org/html/2510.24887v4). Por padrão, apenas `layer4` e `fc` aprendem; as outras camadas e suas estatísticas de BatchNorm ficam congeladas. As camadas liberadas podem ser alteradas em `model_options.trainable_layers`. A LSTM aprende todos os seus pesos e recebe sequências de 64 frames no preset.

Selecionar checkpoints por F1-macro, congelar camadas iniciais e usar LSTM ou busca são escolhas deste projeto. Elas modificam o experimento original, que seleciona checkpoints por acurácia e usa loss no early stopping.
