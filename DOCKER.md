# Imagem Docker para o cluster

A imagem instala Python 3.12.15 e o projeto dentro do container, sem venv. O padrão é PyTorch 2.6.0 e torchvision 0.21.0 com CUDA 12.4, escolhido para o nó A100 com driver 550.54.15. Dados, caches processados, resultados e pesos ficam em volumes separados; não entram no contexto de build.

## 1. Construir

Execute na raiz do projeto, em uma máquina com Docker disponível e acesso à internet. Estes comandos são para o host que executa Docker, não para o shell do container atual. Não é necessário instalar Python ou alterar contas no host.

```bash
sudo docker build -t libras-translator:cu124 .
```

A versão CUDA precisa ser compatível com o driver e a GPU do nó. Antes de escolher a variante para o cluster, consulte `nvidia-smi` no nó com GPU. A indicação de CUDA nesse comando descreve a capacidade do driver; não confirma a versão instalada dentro do container. A imagem já traz as bibliotecas CUDA do PyTorch; o host precisa ter driver NVIDIA e suporte existente a containers com GPU.

Para trocar a variante, mantenha um par compatível de versões de PyTorch e torchvision e selecione o índice oficial de wheels. Exemplo para uma imagem de CPU:

```bash
docker build -t libras-translator:cpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
```

Por padrão, o processo executa como **UID 1004 e GID 1030**, dono da pasta NFS que você mostrou. Isso não altera usuários nem permissões do host. Para outro proprietário, use `--build-arg APP_UID=... --build-arg APP_GID=...`, ou `--user UID:GID` ao executar. Criar a imagem e executá-la como esse usuário não concede permissões adicionais no NFS.

## 2. Conferir a imagem

```bash
docker run --rm libras-translator:cu124 libras-translator --list-models
docker run --rm --gpus all --entrypoint python libras-translator:cu124 \
  -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "GPU indisponível"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
```

O segundo comando confirma que a GPU está disponível e executa uma operação nela. Para uma imagem CPU, omita `--gpus all` e faça essa conferência com `--entrypoint python ... -c 'import torch; print(torch.__version__)'`.

## 3. Container interativo no cluster

Depois de construir ou importar a imagem, execute **no host do cluster**. O comando segue o formato `docker run [OPTIONS] IMAGE [COMMAND]`, com os volumes que você já usa:

```bash
sudo docker run --gpus all \
  --name cluster01_primo04 \
  --shm-size=2g \
  --user 1004:1030 \
  -v /home/primo/src:/home/src \
  -v /data/primo:/home/data \
  -v libras_primo04_runs:/app/runs \
  -v libras_primo04_weights:/app/weights \
  -w /home/src/mhab/models \
  -it libras-translator:cu124 /bin/bash
```

`-w` define a pasta inicial do projeto; ajuste se ela estiver em outro caminho. A imagem aceita `/bin/bash` diretamente. O prompt usa o usuário `libras`, com as permissões do dono da pasta NFS; as dependências já estão instaladas.

Os dois volumes `libras_primo04_*` são criados pelo Docker e preservam resultados e pesos ao recriar o container. Confirme que o armazenamento do Docker está no disco local do nó, pois o SQLite do Optuna não deve ficar em NFS. Os volumes `/home/src` e `/home/data` mantêm os dados do cluster acessíveis.

Dentro do container:

```bash
python --version
libras-translator --list-models
libras-translator --train --config configs/resnet18.toml \
  --dataset /home/src/mhab/models/data/minds_libras \
  --test-signer-id 05 --device cuda:0 --num-workers 4 \
  --output-dir /app/runs --weights-dir /app/weights
```

Para liberar o nome de um container existente, sem apagar seu estado, pare-o quando não houver um treino em andamento e renomeie-o antes de executar o novo `docker run`:

```bash
sudo docker stop cluster01_primo04
sudo docker rename cluster01_primo04 cluster01_primo04_anterior
```

Escolha outro sufixo se esse nome de backup já existir. Para voltar ao novo container depois de sair do shell:

```bash
sudo docker start -ai cluster01_primo04
```

Os volumes novos não importam automaticamente arquivos do container antigo. Antes de retomar uma busca anterior, copie sua pasta completa de resultados para o novo volume. Para preservar os resultados em NFS após encerrar o treino, copie-os de `/app/runs/` para `/home/src/mhab/models/runs/` com Python ou uma ferramenta disponível no cluster.

## 4. Treino direto com pastas de scratch

No host, ajuste os dois caminhos abaixo. `LIBRAS_PROJECT` é o caminho do projeto **no host**, que pode ser diferente de `/home/src/mhab/models` dentro do container antigo. `LIBRAS_SCRATCH` deve estar em disco local do nó, gravável pelo UID 1004; não use NFS para o banco SQLite do Optuna.

```bash
LIBRAS_PROJECT=/home/primo/src/mhab/models
LIBRAS_SCRATCH=/scratch/libras-mhab
mkdir -p "$LIBRAS_SCRATCH/runs" "$LIBRAS_SCRATCH/weights"

docker run --rm --gpus all --shm-size=2g \
  --user 1004:1030 \
  --mount "type=bind,source=$LIBRAS_PROJECT/data,target=/app/data" \
  --mount "type=bind,source=$LIBRAS_SCRATCH/runs,target=/app/runs" \
  --mount "type=bind,source=$LIBRAS_SCRATCH/weights,target=/app/weights" \
  libras-translator:cu124 libras-translator \
  --train --config configs/resnet18.toml \
  --test-signer-id 05 --device cuda:0 --num-workers 4 \
  --output-dir /app/runs --weights-dir /app/weights
```

Crie as pastas como o usuário que tem acesso ao NFS e ao scratch. O processo também precisa escrever em `data/`: o pipeline salva o índice dos landmarks e os caches de processamento, mesmo quando reaproveita os CSVs. Não monte essa pasta como somente leitura.

Os pesos ImageNet são baixados no primeiro uso. Se o job não tiver internet, copie os pesos existentes para `$LIBRAS_SCRATCH/weights/checkpoints/` antes de iniciá-lo. O `--shm-size=2g` reserva memória compartilhada para o carregamento por workers; ajuste conforme os limites do nó.

O preset atual usa **200 tentativas Optuna**, com validação por pessoa. Para verificar o funcionamento antes dessa busca longa, use um diretório separado e acrescente `--search-trials 1 --validation-signer-id 01`. Isso ainda executa um treinamento interno, o final e o teste; não use seus resultados como seleção definitiva.

Ao retomar, use a mesma imagem, os mesmos volumes e argumentos. O pipeline pode continuar a busca e os treinos interrompidos. O conteúdo de scratch persiste após remover o container, mas o cluster pode apagar esse disco entre jobs. Copie os resultados para armazenamento persistente quando o processo estiver parado; para retomada em outro nó, restaure a pasta inteira antes de iniciar.

```bash
mkdir -p "$LIBRAS_PROJECT/runs" "$LIBRAS_PROJECT/weights"
rsync -a "$LIBRAS_SCRATCH/runs/" "$LIBRAS_PROJECT/runs/"
rsync -a "$LIBRAS_SCRATCH/weights/" "$LIBRAS_PROJECT/weights/"
```

Use uma execução por busca/GPU. Não execute dois processos escrevendo nos mesmos resultados, banco ou caches de dados.

## Configurações e dependências

O código e os presets são copiados para a imagem. Após alterar o código, reconstrua a imagem. Para usar configurações atualizadas sem reconstruir, acrescente este volume ao `docker run`:

```bash
--mount "type=bind,source=$LIBRAS_PROJECT/configs,target=/app/configs,readonly"
```

Python, PyTorch, torchvision e MediaPipe têm versões fixas. As outras dependências seguem os intervalos do `pyproject.toml`, portanto reconstruções futuras podem resolvê-las de forma diferente. A imagem registra as versões instaladas em `/app/requirements-installed.txt`:

```bash
docker run --rm --entrypoint cat libras-translator:cu124 /app/requirements-installed.txt
```

Para preservar exatamente o ambiente de um experimento, mantenha a imagem usada e registre seu ID junto com os resultados.

## Transferir a imagem pronta

Se você construir em outra máquina, pode transferir a imagem uma única vez, separada dos dados. A imagem CUDA pode ocupar vários GB.

```bash
docker save -o /tmp/libras-translator-cu124.tar libras-translator:cu124
rsync -avhP -e "ssh -J mhab@192.168.155.9" \
  /tmp/libras-translator-cu124.tar \
  primo@192.168.155.1:/home/primo/src/mhab/
```

No host de destino com Docker disponível:

```bash
docker load -i /home/primo/src/mhab/libras-translator-cu124.tar
```

Nesta estação, o build pode ser feito com `podman build -t libras-translator:cu124 .`. Para transferir essa imagem ao Docker do cluster, exporte no formato Docker:

```bash
podman save --format docker-archive \
  -o /tmp/libras-translator-cu124.tar localhost/libras-translator:cu124
```

Após o `docker load` no destino, ajuste o nome importado:

```bash
docker tag localhost/libras-translator:cu124 libras-translator:cu124
```

Fontes: [imagem oficial Python](https://hub.docker.com/_/python), [pares oficiais PyTorch/torchvision e variantes CUDA](https://pytorch.org/get-started/previous-versions/), [GPU em containers NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) e [restrições do SQLite no Optuna](https://optuna.readthedocs.io/en/stable/faq.html).
