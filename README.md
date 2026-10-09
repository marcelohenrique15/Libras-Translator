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
libras-translator --train --config configs/resnet18.toml
libras-translator --list-models
```

O preset ResNet18 já inclui a busca Optuna de `configs/search.toml` e reserva o sinalizador **05** para teste. O protocolo é:

1. Preparar manifesto, landmarks, seleção, interpolação, ancoragem e codificação, reutilizando caches atuais.
2. Buscar hiperparâmetros somente nos sinalizadores disponíveis para treinamento. Cada tentativa alterna a pessoa de validação; o teste externo permanece reservado.
3. Escolher a configuração com maior **F1 macro médio de validação**. A mediana das melhores épocas de suas divisões define a duração do treinamento final, arredondando `.5` para cima.
4. Inicializar **um modelo novo** e treiná-lo em todos os sinalizadores fora do teste pelo número de épocas escolhido.
5. Avaliar esse modelo no teste reservado e gerar métricas, tabelas e gráficos.

Com os 12 sinalizadores atuais e 200 tentativas no espaço de busca para cluster, são **2.200 treinamentos internos e um treinamento final** para o teste 05. O final usa os 11 sinalizadores disponíveis, sem validação ou early stopping: a duração já foi definida na etapa anterior. Não se escolhem configurações, épocas ou checkpoints pelo resultado do teste.

`--dataset` aceita uma pasta em `data/` ou um caminho. A forma `--train data/minds_libras` também funciona. Argumentos CLI substituem o TOML ou JSON carregado; opções específicas ficam em `model_options` e `processor_options`.

O experimento usa processamento determinístico, **sem data augmentation**, em treino, validação e teste. `augmentation=true` é rejeitado. O preset procura reduzir overfitting com congelamento de camadas, dropout, AdamW, penalidades L1/L2 e label smoothing; essas técnicas não garantem ganho de generalização.

| Opções | Uso |
|---|---|
| `--epochs`, `--patience` | Limite de épocas e early stopping nas validações internas |
| `--final-epochs` | Duração final já selecionada, para uma configuração sem busca |
| `--batch-size`, `--learning-rate`, `--weight-decay` | Treino; weight decay desacoplado do AdamW |
| `--l1-lambda`, `--l2-lambda`, `--label-smoothing`, `--gradient-clip` | Penalidades explícitas, suavização de rótulos e clipping |
| `--test-signer-id ID\|all`, `--validation-signer-id ID` | Divisão por sinalizador |
| `--subset asl_2nd\|all\|arcanjo`, `--frame-count` | Pontos e tamanho das sequências |
| `--anchor shoulders\|nose\|none` | Referência fixa por vídeo; padrão: centro dos ombros |
| `--imputation` / `--no-imputation` | Processamento |
| `--processor NOME` | Processor registrado ou `pacote.modulo:Classe` |
| `--device auto\|cpu\|cuda:N`, `--num-workers`, `--seed` | Execução |
| `--output-dir`, `--weights-dir` | Resultados e pesos compartilhados |
| `--search-config`, `--search-trials` | Espaço de busca Optuna e total de tentativas concluídas |
| `--reuse-search`, `--force-restart` | Importar uma busca salva ou iniciar treinos em um diretório novo |

Sem escolher o teste, usa o sinalizador marcado no manifesto ou o primeiro ID disponível. Cada outro sinalizador é usado para validação. `--validation-signer-id` restringe a uma rodada. Para LOPO completo:

```bash
libras-translator --train --config configs/resnet18.toml --test-signer-id all
```

Cada teste externo tem um estudo próprio. Com 12 sinalizadores e 200 tentativas por estudo, o LOPO completo custa **26.400 treinamentos internos + 12 finais = 26.412 treinamentos**. `--validation-signer-id` reduz a seleção a uma única pessoa de validação; o modelo final ainda treina em todas as pessoas fora do teste.

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

### Coordenadas relativas e codificação

O processor de landmarks usa `anchor = "shoulders"` por padrão. Após a interpolação opcional, calcula o ponto médio dos ombros em cada frame válido e usa a mediana desses centros como referência fixa do vídeo. Subtrai essa referência de todos os pontos, incluindo mãos e rosto, preservando os valores ausentes. Isso remove a posição média no enquadramento e mantém as trajetórias; não corrige escala ou rotação nem acompanha deslocamentos da pessoa durante o vídeo.

Escolha a referência em `processor_options`:

```toml
[processor_options]
anchor = "shoulders" # Centro dos ombros; padrão
# anchor = "nose"   # Mediana da posição do nariz
# anchor = "none"   # Coordenadas originais; processamento legado
```

Na representação de imagem, as coordenadas relativas usam o intervalo fixo `[-1, 1]`, com corte dos valores fora dele. Valores válidos viram intensidades de 1 a 255: a coordenada relativa zero vira 128, cinza médio. Ausências viram 0, preto. O encoder mantém três frames consecutivos nos canais RGB e redimensiona a imagem para 224 × 224; esse redimensionamento mistura cores vizinhas e não fornece uma máscara explícita de ausência. Na representação de sequência, as coordenadas continuam relativas e as ausências são preenchidas com zero antes da reamostragem temporal.

A ancoragem exige **novo treinamento**. Os CSVs da extração continuam aproveitáveis; os caches das etapas seguintes têm identificação por código e opções, preservando os arquivos antigos. Para reproduzir o processamento de um modelo anterior, use explicitamente `anchor = "none"`.

`prepare(sample_id)` retorna os pontos interpolados e ancorados, ainda com `NaN` nas ausências; `encode(prepared)` transforma esses pontos na entrada do modelo. `process(sample_id)` executa os dois passos e reutiliza o cache. Se não houver nenhum frame com uma referência completa, a preparação informa o erro e o ID da amostra.

## Busca de hiperparâmetros

```bash
libras-translator --train --config configs/resnet18.toml --search-trials 10
```

O Optuna usa TPE e maximiza a média do F1 macro nas validações internas. O espaço padrão busca taxa de aprendizado, batch, limite de épocas, weight decay do AdamW, L1, L2, label smoothing, largura e quantidade de camadas do classificador, dropout e quais camadas da ResNet podem aprender. `epochs` limita cada treino interno; a duração final deriva das melhores épocas da configuração vencedora.

O espaço para cluster usa 200 tentativas, com 30 iniciais aleatórias. Explora cabeças de 1–4 camadas e 32–1.024 unidades, dropout de 0–0,7 e cinco opções de congelamento, incluindo liberar toda a ResNet. Os limites de épocas vão de 30 a 300 e a patience de 5 a 20. O comando acima, limitado a 10 tentativas, serve para uma verificação curta e fica inteiramente na fase aleatória. A busca continua sequencial, mesmo no cluster.

O console mostra o protocolo por sinalizador, o orçamento e espaço da busca, a configuração completa antes de cada tentativa (inclusive ao retomar) e a configuração vencedora. O limite interno de épocas impresso na busca é convertido na duração final pela regra da mediana.

Repetir o mesmo comando reutiliza o estudo, as divisões concluídas, o treinamento final e o teste quando ainda correspondem ao código, aos dados e à configuração. Treinos interrompidos retomam a partir de `last.pt`; uma época incompleta é repetida. O orçamento representa o total de tentativas **concluídas**, não novas tentativas a cada execução. Para ampliar a busca para trezentas no total:

```bash
libras-translator --train --config configs/resnet18.toml \
  --search-trials 300
```

Para executar uma nova busca e novos treinos sem apagar resultados anteriores:

```bash
libras-translator --train --config configs/resnet18.toml --force-restart
```

`--force-restart` cria um diretório `fresh_<id>` e preserva os caches de pré-processamento. Quando `pretrained=true`, a inicialização nova usa pesos ImageNet e um classificador novo; não reaproveita pesos aprendidos nas divisões da busca.

Para aproveitar apenas os parâmetros vencedores de uma busca salva e fazer um novo treinamento final:

```bash
libras-translator --train --config configs/resnet18.toml \
  --reuse-search runs/minds_libras/search_1cf0ec14bae0 --force-restart
```

Essa busca legada seleciona nove épocas finais. Ela usou outro protocolo: agora a augmentation está desativada e o otimizador é AdamW, além das opções novas de regularização. Seu F1 antigo descreve as validações de origem e **não valida o protocolo atual**. A importação registra essa origem; recomendamos a nova busca padrão para selecionar parâmetros no experimento atual.

Também é possível importar hiperparâmetros de uma busca anterior à ancoragem. Nesse caso, o treino final usa a referência da configuração atual, e o resultado registra a mudança de processamento; o F1 da busca de origem não valida o modelo ancorado.

O arquivo `best_config_test<ID>.json` já contém `final_epochs` e desativa a busca. Carregá-lo executa diretamente o treinamento final e o teste; use `--force-restart` para treinar um novo modelo mesmo se essa execução já existir:

```bash
libras-translator --train --config caminho/best_config_test05.json --force-restart
```

Defina opções e intervalos no [arquivo de busca](configs/search.toml). O [guia de extensão](docs/EXTENDING.md#configuração-e-busca) explica como aplicá-los a outras arquiteturas e processors.

## Resultados

O console informa o caminho de `summary.json`. Cada teste externo produz um diretório `final/test_<ID>/<execução>/` com:

| Arquivo | Conteúdo |
|---|---|
| `final.pt`, `last.pt` | Modelo final e estado para retomada |
| `config.json`, `training.json`, `result.json`, `test.json` | Configuração, seleção de épocas, origem da busca e métricas |
| `history.json` | Épocas do treinamento final |
| `predictions.csv`, `per_class.csv` | Previsões com confiança softmax e métricas por classe |
| `graphs/loss.png`, `graphs/f1.png` | Cross-entropy, objetivo regularizado e F1 do treino final |
| `graphs/validation_loss.png`, `graphs/validation_f1_macro.png` | Curvas internas da configuração vencedora, quando disponíveis |
| `graphs/search.png` | Evolução da busca Optuna, quando disponível |
| `graphs/confusion_matrix*.png`, `graphs/per_class.png` | Matriz bruta, matriz normalizada por classe real e precisão/recall/F1 |

As curvas finais não têm validação: o modelo recebe todos os dados de desenvolvimento. As curvas de validação pertencem aos treinamentos internos da busca. A confiança softmax é a probabilidade atribuída pelo modelo à classe prevista; não implica calibração de confiança.

## Organização e extensão

```text
src/cli/              # argumentos e chamada das etapas
src/dataset/          # manifesto, divisão e carregamento
src/preprocessor/     # extração, seleção, interpolação, ancoragem e codificação
src/models/           # arquiteturas e registry
src/training/         # configuração, seleção, treino, busca, relatórios e pipeline
configs/              # presets e espaços de busca
data/<dataset>/       # vídeos, manifesto e entradas processadas
runs/<dataset>/       # configurações, checkpoints e métricas
weights/checkpoints/  # pesos ImageNet compartilhados
tests/                # verificações do pipeline
```

Para adicionar modelos ou processors sem editar o pipeline, veja [como estender o projeto](docs/EXTENDING.md).

## Cluster

Para construir `cluster_docker_image:latest` e recriar `cluster01_primo07` com as dependências do `pyproject.toml`, veja o [guia Docker](DOCKER.md). A imagem usa Python 3.12 e dispensa venv. Apenas `/home/primo/src/mhab/models` é montada no container; dados, caches, resultados e pesos ficam dentro do projeto.

Prepare os dados e os pesos em um ambiente com acesso à rede antes do job. Os pesos ImageNet são baixados no primeiro uso; também podem ser copiados para `weights/checkpoints/`.

Para treinar sem transferir os vídeos, copie `metadata/manifest.csv`, os CSVs de landmarks e `processed/landmarks/index.json`. Preserve as datas dos CSVs; o índice confirma a extração completa sem os vídeos originais.

```bash
libras-translator --encode-landmarks --config configs/resnet18.toml
libras-translator --train --config configs/resnet18.toml \
  --test-signer-id 05 \
  --device cuda:0 --num-workers 4 \
  --output-dir /home/src/mhab/models/runs \
  --weights-dir /home/src/mhab/models/weights
```

Cada processo deve escrever em uma execução exclusiva. O projeto executa um processo por GPU; não distribui um treino entre GPUs nem envia jobs ao Slurm. `cuda:0` considera as GPUs visíveis para o processo.

A busca Optuna executa tentativas em sequência. Use um processo por busca e não compartilhe o mesmo banco entre processos. No cluster informado, a pasta do projeto está em NFS; o SQLite do Optuna nessa pasta tem limitações de bloqueio, descritas no [guia Docker](DOCKER.md#banco-optuna-no-nfs).

## Experimento de referência

O preset ResNet usa ASL-2nd com 80 pontos e interpolação, baseado no [artigo](https://arxiv.org/html/2510.24887v4), e acrescenta a ancoragem fixa no centro dos ombros descrita acima. Na configuração base, apenas `layer4` e `fc` aprendem; as outras camadas e suas estatísticas de BatchNorm ficam congeladas. A busca também considera treinar somente `fc`.

A ResNet termina em softmax para fornecer probabilidades. Durante o treinamento, `forward_logits()` entrega os valores anteriores ao softmax à cross-entropy, que calcula log-softmax de forma numericamente estável. O objetivo inclui label smoothing e as penalidades L1/L2 escolhidas; a loss registrada para comparação com a validação é a cross-entropy sem suavização ou penalidades.

Seleção por F1 macro, busca, congelamento de camadas e treinamento final são escolhas deste projeto e devem ser descritos no relatório experimental. L1/L2 explícitas e weight decay do AdamW são controles diferentes, aplicados aos pesos treináveis; bias e parâmetros de normalização não são penalizados.
