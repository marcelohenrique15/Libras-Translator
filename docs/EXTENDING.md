# Como estender o projeto

## Modelo novo

Crie uma classe em `src/models/` que herde `torch.nn.Module`, declare o formato de entrada e retorne logits. `class_count` é definido pelo manifesto:

```python
from torch import nn


class MyCNN(nn.Module):
    INPUT_FORMAT = "image"

    def __init__(self, class_count: int, channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(channels, class_count)

    def forward(self, images):
        features = self.features(images).flatten(start_dim=1)
        return self.classifier(features)
```

Salvando em `src/models/my_cnn.py`, use diretamente:

```bash
libras-translator --train --dataset minds_libras --model models.my_cnn:MyCNN
```

Ou importe a classe e acrescente uma entrada em `ModelRegistry.MODELS` para usar um nome curto. O trainer e o pipeline não precisam mudar.

| `INPUT_FORMAT` | Entrada do modelo |
|---|---|
| `image` | `(lote, 3, altura, largura)` |
| `sequence` | `(lote, frames, coordenadas)` |

`forward()` retorna `(lote, class_count)`. Não aplique softmax: cross-entropy recebe logits. Parâmetros específicos vão em `[model_options]` no TOML:

```toml
model = "models.my_cnn:MyCNN"

[model_options]
channels = 64
```

Para modelos temporais, declare `INPUT_FORMAT = "sequence"`. O processor padrão reamostra o vídeo para `frame_count` frames e mantém `x,y` de cada ponto. Outros formatos podem usar outro nome em `INPUT_FORMAT` e um processor próprio.

## Processor novo

Crie uma classe cujo construtor receba `dataset_root` e as opções do processamento. Use pelo CLI `--processor pacote.modulo:Classe` ou acrescente uma entrada em `ProcessorRegistry.PROCESSORS`.

Contrato público:

| Método | Responsabilidade |
|---|---|
| `prepare(sample_id)` | Preparar uma amostra antes da augmentation |
| `augment(prepared)` | Transformar somente a entrada de treino |
| `encode(prepared)` | Retornar a entrada do modelo como `numpy.float32` |
| `process(sample_id)` | Preparar e codificar sem augmentation; pode usar cache |
| `prepare_all(rows, stage="encode")` | Executar uma etapa para todo o manifesto |

No treino, o Dataset chama `prepare → augment → encode`. Na validação e no teste, chama `process`. A augmentation deve preservar a amostra armazenada. O processor padrão recebe `representation="image"` ou `"sequence"` conforme o modelo. Processors personalizados recebem as opções de `[processor_options]` e executam suas próprias etapas necessárias, incluindo extração quando aplicável. Processors temporais podem expor `input_size` para informar a quantidade de coordenadas aos modelos que usam esse argumento.

Para os comandos por etapa, `prepare_all` recebe `stage="select"`, `"impute"` ou `"encode"`. Cada processor define o conteúdo dessas etapas. Suas opções ficam em `[processor_options]`.

O processor padrão também expõe `select(sample_id)` e `impute(sample_id)`; essas funções ajudam a modificar os passos separadamente.

## Configuração e busca

Copie um preset de `configs/`. Os campos gerais estão em `TrainingConfig`; opções de arquiteturas e processors ficam nas respectivas tabelas. Argumentos CLI substituem a configuração carregada, sem modificar o arquivo.

O espaço de busca Optuna é outro TOML. Listas representam opções categóricas; tabelas definem intervalos contínuos ou inteiros:

```toml
n_trials = 20

[parameters]
learning_rate = { type = "float", low = 0.00001, high = 0.001, log = true }
batch_size = [32, 64]

[parameters.model_options]
channels = [32, 64]

[parameters.processor_options]
frame_count = { type = "int", low = 32, high = 96, step = 16 }
```

Use apenas opções aceitas pelo modelo e processor escolhidos. Os campos gerais ajustáveis são `learning_rate`, `weight_decay`, `batch_size`, `epochs` e `patience`. Modelo, processor, dataset e divisões permanecem fixos no estudo.

O TPE sugere configurações usando o histórico e o estudo usa `direction="maximize"`: o objetivo é a média do **F1-macro de validação**. Todas as classes do manifesto entram na média, inclusive classes sem acertos na rodada. O teste fica reservado até a seleção da melhor tentativa. Não há garantia de melhora a cada tentativa.

```bash
libras-translator --train --config configs/meu_modelo.toml --search-config configs/minha_busca.toml
```

`n_trials` é o total de tentativas concluídas por sinalizador externo; `--search-trials 40` substitui esse total. Aumentar o orçamento continua o mesmo estudo. Alterar dados, código, espaço ou divisão cria outra busca. O histórico, parâmetros e métricas são persistidos em `optuna.sqlite3`; uma tentativa interrompida reutiliza sua configuração e seus checkpoints.

## Arquivos e retomada

Código, dados ou configurações que afetam um resultado identificam uma nova execução; caches de etapas anteriores são reutilizados quando continuam atuais. `last.pt` preserva época, otimizador e geradores aleatórios. A retomada começa na próxima época; uma época interrompida é repetida.

Ao executar jobs separados, use combinações exclusivas de teste e validação. Não execute dois processos escrevendo na mesma rodada.

A busca é sequencial: um processo por busca, com o SQLite em armazenamento local do nó. Não use o mesmo banco em buscas simultâneas nem armazene o banco em NFS.
