# Docker no cluster

A imagem `cluster_docker_image:latest` instala o projeto e suas dependências do `pyproject.toml`, sem venv. Ela usa Python 3.12.15, PyTorch 2.6.0 e torchvision 0.21.0 com CUDA 12.4.

O container `cluster01_primo07` monta apenas a pasta do projeto:

| Local | Caminho |
|---|---|
| Host do cluster | `/home/primo/src/mhab/models` |
| Dentro do container | `/home/src/mhab/models` |

Dados, caches, resultados e pesos ficam nessa pasta. O build não inclui esses arquivos; eles são acessados pela montagem do projeto. Não são necessários volumes nomeados nem pastas de trabalho externas.

## 1. Construir a imagem atualizada

Execute **no host do cluster, fora do container**, depois de enviar o projeto atualizado:

```bash
cd /home/primo/src/mhab/models && \
sudo docker build -t cluster_docker_image:latest .
```

O Docker instala as bibliotecas declaradas no `pyproject.toml`. O build verifica as dependências com `pip check` e testa os imports necessários. Um conflito com as versões fixadas de PyTorch ou torchvision interrompe o build.

Para adicionar uma biblioteca, declare-a no `pyproject.toml`, envie o arquivo atualizado e reconstrua a imagem. Bibliotecas instaladas manualmente em um container antigo não são incorporadas automaticamente à nova imagem.

O usuário da imagem tem **UID 1004 e GID 1030**, compatíveis com a pasta do projeto no NFS. A criação da imagem não altera usuários, permissões ou instalações de Python no host.

## 2. Testar antes de substituir o container

Ainda no host, confira as dependências e a GPU na imagem nova:

```bash
sudo docker run --rm --gpus all --shm-size=2g \
  --user 1004:1030 \
  --mount type=bind,source=/home/primo/src/mhab/models,target=/home/src/mhab/models \
  -w /home/src/mhab/models \
  cluster_docker_image:latest \
  /bin/bash -c "python -m pip check && python -m cli.main --list-models && python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.version.cuda); assert torch.cuda.is_available(), \"GPU indisponível\"; print(\"GPU:\", torch.cuda.get_device_name(0)); print(torch.ones(1, device=\"cuda\") + 1)'"
```

Esse container de teste é removido automaticamente ao terminar. Se o comando falhar, corrija o problema antes de substituir `cluster01_primo07`.

## 3. Recriar somente `cluster01_primo07`

Antes de remover o container antigo, termine ou interrompa o treinamento e confira suas montagens:

```bash
sudo docker inspect cluster01_primo07 \
  --format '{{range .Mounts}}{{println .Type .Source "->" .Destination}}{{end}}'
```

Arquivos da pasta montada do projeto permanecem no host. Arquivos salvos apenas na camada interna do container serão perdidos na remoção; copie qualquer resultado necessário para `models/` antes de continuar, incluindo resultados antigos que ainda estejam em `/app`.

Depois do teste da imagem e da conferência dos arquivos, execute no host:

```bash
sudo docker stop cluster01_primo07 && \
sudo docker rm cluster01_primo07 && \
sudo docker run --gpus all \
  --name cluster01_primo07 \
  --shm-size=2g \
  --user 1004:1030 \
  --mount type=bind,source=/home/primo/src/mhab/models,target=/home/src/mhab/models \
  -w /home/src/mhab/models \
  -it cluster_docker_image:latest /bin/bash
```

O comando remove somente o container indicado, sem `-v`, e recria o mesmo nome. O novo container inicia na pasta do projeto com as dependências já instaladas.

Para voltar ao container depois de sair:

```bash
sudo docker start -ai cluster01_primo07
```

Se ele já estiver rodando, abra outro shell:

```bash
sudo docker exec -it -w /home/src/mhab/models cluster01_primo07 /bin/bash
```

## 4. Pré-processar e treinar

Execute **dentro do container**. A seleção, a interpolação, a ancoragem e a codificação usam CPU:

```bash
python -m cli.main \
  --encode-landmarks \
  --config configs/resnet18.toml \
  --dataset /home/src/mhab/models/data/minds_libras \
  --anchor shoulders \
  --imputation \
  --device cpu
```

Para uma nova busca de hiperparâmetros seguida de treinamento final, teste e gráficos:

```bash
python -m cli.main \
  --train \
  --config configs/resnet18.toml \
  --dataset /home/src/mhab/models/data/minds_libras \
  --anchor shoulders \
  --imputation \
  --test-signer-id 05 \
  --device cuda:0 \
  --num-workers 4 \
  --search-trials 200 \
  --output-dir /home/src/mhab/models/runs \
  --weights-dir /home/src/mhab/models/weights \
  --force-restart
```

O treinamento também prepara os dados automaticamente. Caches válidos são reutilizados. `--force-restart` cria uma nova execução de busca e treinamento, preservando resultados anteriores e os caches de processamento. Com `pretrained = true`, os modelos começam dos pesos ImageNet.

Cada uso de `--force-restart` cria outro diretório. Se precisar retomar uma execução, comece com um `--output-dir` exclusivo e sem essa opção; depois repita exatamente o mesmo comando. Não execute dois processos escrevendo no mesmo estudo ou nos mesmos caches.

Os pesos iniciais podem ser enviados previamente para `weights/checkpoints/`; caso contrário, serão baixados no primeiro uso. O processor precisa de acesso de escrita à pasta `data/` para salvar seus caches.

## Código, caches e ambiente

A imagem define `PYTHONPATH=/home/src/mhab/models/src`. Assim, ao montar o projeto, o Python usa o código atualizado do host. Alterações em código e configurações ficam disponíveis para a próxima execução; alterações em dependências exigem reconstruir a imagem.

O diretório inicial e o diretório pessoal do usuário `libras` apontam para `/home/src/mhab/models`. Os caches do Matplotlib, CUDA e bibliotecas que usam XDG ficam em `.cache/` dentro do projeto. Os resultados de treino usam `runs/`, e os pesos compartilhados usam `weights/`. Dependências do Python são instaladas em `/usr/local` dentro da imagem.

O build registra as versões instaladas em `/home/src/mhab/models/requirements-installed.txt`. Como a montagem do projeto cobre esse arquivo no container de trabalho, consulte o registro diretamente na imagem, sem a montagem:

```bash
sudo docker run --rm --entrypoint cat \
  cluster_docker_image:latest \
  /home/src/mhab/models/requirements-installed.txt
```

As dependências com intervalos no `pyproject.toml` podem resolver versões diferentes em reconstruções futuras. Registre o ID da imagem usada no experimento:

```bash
sudo docker image inspect cluster_docker_image:latest --format '{{.Id}}'
```

### Banco Optuna no NFS

No cluster informado, `models/` está em NFS. Portanto, o banco SQLite do Optuna salvo em `runs/` também ficará no NFS. O Optuna desaconselha essa combinação por limitações de bloqueio de arquivos. Manter uma execução por estudo evita concorrência entre processos, mas não elimina a limitação do NFS; um backend de banco compatível exige configuração adicional no pipeline.

Fontes: [imagem oficial Python](https://hub.docker.com/_/python), [pares oficiais PyTorch/torchvision](https://pytorch.org/get-started/previous-versions/), [montagens de pastas no Docker](https://docs.docker.com/engine/storage/bind-mounts/), [GPU em containers NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) e [restrições do SQLite no Optuna](https://optuna.readthedocs.io/en/stable/faq.html).
