# Como estender o projeto

## Modelo novo

Crie uma classe em `src/models/` que herde `torch.nn.Module`, declare o formato de entrada e receba `class_count`, definido pelo manifesto. O formato de saída padrão é logits:

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

`forward()` retorna `(lote, class_count)`. Sem `OUTPUT_FORMAT`, ou com `OUTPUT_FORMAT = "logits"`, o trainer utiliza essa saída na cross-entropy e calcula softmax para as probabilidades de avaliação.

Para expor probabilidades diretamente, use o mesmo contrato da ResNet18: declare `OUTPUT_FORMAT = "probabilities"`, implemente `forward_logits()` e aplique softmax apenas em `forward()`. O trainer chama `forward_logits()` durante treino e avaliação, mantendo a loss estável:

```python
OUTPUT_FORMAT = "probabilities"

def forward_logits(self, images):
    features = self.features(images).flatten(start_dim=1)
    return self.classifier(features)

def forward(self, images):
    return self.forward_logits(images).softmax(dim=1)
```

Esses métodos pertencem à classe do modelo. Não passe probabilidades à `CrossEntropyLoss`; ela aplica log-softmax internamente. Parâmetros específicos vão em `[model_options]` no TOML:

```toml
model = "models.my_cnn:MyCNN"

[model_options]
channels = 64
```

Para modelos temporais, declare `INPUT_FORMAT = "sequence"`. O processor padrão reamostra o vídeo para `frame_count` frames e mantém `x,y` de cada ponto. Outros formatos podem usar outro nome em `INPUT_FORMAT` e um processor próprio.

## Processor novo

Crie uma classe cujo construtor receba `dataset_root` e as opções do processamento. Use pelo CLI `--processor pacote.modulo:Classe` ou acrescente uma entrada em `ProcessorRegistry.PROCESSORS`.

Contrato necessário para o pipeline:

| Método | Responsabilidade |
|---|---|
| `process(sample_id)` | Retornar a entrada do modelo como `numpy.float32`; pode usar cache |
| `prepare_all(rows, stage="encode")` | Executar uma etapa para todo o manifesto |

O Dataset chama `process()` em treino, validação e teste. O processamento deve ser determinístico e não usar augmentation; `TrainingConfig` rejeita `augmentation=true`. O processor padrão recebe `representation="image"` ou `"sequence"` conforme o modelo. Processors personalizados recebem `[processor_options]` e executam suas próprias etapas, incluindo extração quando aplicável. Processors temporais podem expor `input_size` para informar a quantidade de coordenadas aos modelos que usam esse argumento.

Para os comandos por etapa, `prepare_all` recebe `stage="select"`, `"impute"` ou `"encode"`. Cada processor define o conteúdo dessas etapas. Suas opções ficam em `[processor_options]`.

`prepare_all()` ocorre antes das divisões de treino, validação e teste. Use apenas transformações determinísticas por amostra, como normalização pelas medidas do próprio vídeo. Não ajuste média/desvio globais, PCA ou outras transformações aprendidas com todo o manifesto: isso vazaria informação entre divisões. Esses recursos exigem um contrato adicional que ajuste a transformação somente no treino de cada divisão, ainda não implementado.

O processor padrão também expõe `prepare(sample_id)`, `encode(prepared)`, `select(sample_id)` e `impute(sample_id)` para modificar os passos separadamente. Um processor novo pode organizar seus métodos internos de outra forma, desde que cumpra o contrato acima.

## Configuração e busca

Copie `configs/resnet18.toml` e adapte `model`, `model_options` e o espaço de busca ao novo modelo. Os campos gerais estão em `TrainingConfig`; opções de arquiteturas e processors ficam nas respectivas tabelas. Argumentos CLI substituem a configuração carregada, sem modificar o arquivo.

O espaço de busca Optuna é outro TOML. Listas representam opções categóricas; tabelas definem intervalos contínuos ou inteiros:

```toml
n_trials = 20

[parameters]
learning_rate = { type = "float", low = 0.00001, high = 0.001, log = true }
batch_size = [32, 64]
epochs = [15, 30]
weight_decay = [0.0, 0.0001, 0.001]
l1_lambda = [0.0, 0.0000001]
l2_lambda = [0.0, 0.000001]
label_smoothing = [0.0, 0.05, 0.1]

[parameters.model_options]
channels = [32, 64]

[parameters.processor_options]
frame_count = { type = "int", low = 32, high = 96, step = 16 }
```

Use somente opções aceitas pelo modelo e processor. Os campos gerais ajustáveis são `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience`, `l1_lambda`, `l2_lambda`, `label_smoothing` e `gradient_clip`. Modelo, processor, dataset e divisões permanecem fixos no estudo. O espaço ResNet também busca `hidden_size`, `num_layers`, `dropout` e `trainable_layers`; essa quantidade de camadas se refere ao classificador, mantendo o backbone ResNet18.

Categorias Optuna devem ser escalares. Para `model_options.trainable_layers`, use strings como `"fc"` e `"layer4,fc"`; a configuração converte cada uma em uma lista de camadas. Outras opções específicas são passadas ao construtor sem adaptação.

O TPE sugere configurações usando o histórico e maximiza o **F1 macro médio de validação**. Todas as classes do manifesto entram na média, inclusive classes sem acertos. Cada tentativa treina em pessoas diferentes das usadas para validação, sempre excluindo a pessoa do teste externo. O maior F1 de validação seleciona `best.pt` e controla o early stopping interno.

A configuração vencedora passa a `ModelSelection`. A mediana de suas melhores épocas, arredondando `.5` para cima, define `final_epochs`; `epochs` na busca é o limite máximo de cada divisão. O pipeline cria um modelo novo, treina todos os sinalizadores fora do teste por essa duração fixa, salva `final.pt` e só então avalia o teste. Esse treino final não contém validação nem seleção de época pelo teste. Com `pretrained=true`, o modelo novo parte novamente dos pesos pré-treinados, sem copiar checkpoints das validações.

O trainer usa AdamW. `weight_decay` é desacoplado do objetivo; `l1_lambda` e `l2_lambda` acrescentam somas de valores absolutos e quadrados dos pesos. As três penalizações excluem parâmetros congelados, bias e normalização. A busca pode combinar penalizações; valores altos podem prejudicar o ajuste, por isso compare usando validação. Label smoothing modifica o objetivo de classificação no treino. Os registros distinguem `train_loss` (cross-entropy comum, comparável à validação) de `train_objective` (suavização e penalidades explícitas).

```bash
libras-translator --train --config configs/meu_modelo.toml --search-config configs/minha_busca.toml
```

`n_trials` é o total de tentativas concluídas por sinalizador externo; `--search-trials 40` substitui esse total. Aumentar o orçamento continua o estudo. O histórico, parâmetros e métricas ficam em `optuna.sqlite3`; uma tentativa interrompida reutiliza sua configuração e seus checkpoints. Não há garantia de melhoria a cada tentativa nem de encontrar um ótimo global.

Sem `search_config` e sem `final_epochs`, o pipeline valida uma configuração fixa internamente para selecionar sua duração. Com `final_epochs` já informado e sem busca, executa diretamente o treinamento final e teste. Os JSONs `best_config_test<ID>.json` gerados pelo pipeline já têm esses campos preparados.

## Arquivos e retomada

Código, dados ou configurações que afetam um resultado identificam uma nova execução; caches de etapas anteriores são reutilizados quando continuam atuais. `last.pt` preserva época, otimizador e geradores aleatórios. A retomada começa na próxima época; uma época interrompida é repetida. Os modos de validação e treino final usam diretórios separados.

`--force-restart` cria um diretório novo `fresh_<id>`, sem apagar artefatos ou reutilizar treinos anteriores. Os caches de pré-processamento continuam disponíveis. Combine com `--reuse-search caminho/da/busca` para importar somente a configuração vencedora e refazer o treinamento final; a importação confere dataset, modelo, processor, divisões e processamento.

Buscas legadas podem ser importadas, preservando os arquivos de origem e registrando a procedência em `result.json`. A importação compara também a assinatura do código e a semente. Mudanças de código, semente, augmentation, otimizador ou regularização, assim como uma origem legada sem assinatura verificável, marcam o score de origem como inválido para o protocolo atual. Os parâmetros ainda podem servir como ponto de partida; para validar essas escolhas, execute uma busca nova.

`ExperimentReporter` recebe o histórico final, métricas de teste, probabilidades por amostra e, opcionalmente, o relatório Optuna e os históricos das validações vencedoras. Ele escreve CSVs e gráficos com Agg, sem interface gráfica. As curvas finais de loss/F1 ficam separadas das curvas internas de validação; matrizes de confusão e métricas por classe usam somente o teste final.

Ao executar jobs separados, use diretórios exclusivos. Não execute dois processos escrevendo na mesma rodada.

A busca é sequencial: um processo por busca, com o SQLite em armazenamento local do nó. Não use o mesmo banco em buscas simultâneas nem armazene o banco em NFS.
