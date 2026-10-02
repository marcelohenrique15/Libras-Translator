# MINDS-Libras

Coloque os vídeos em `raw/`, diretamente ou em subpastas por sinalizador, preservando os nomes originais.

Exemplo: `01AcontecerSinalizador05-3.mp4` representa classe `01` (Acontecer), sinalizador `05` e repetição `3`. O builder MINDS inclui apenas repetições de 1 a 5. A cópia usada no projeto contém 1.155 vídeos de 20 classes e 12 sinalizadores.

Na raiz do repositório:

```bash
libras-translator --build-manifest --dataset minds_libras --dataset-format minds_libras
libras-translator --split-manifest --dataset minds_libras --test-signer-id 05
libras-translator --train --config configs/resnet18.toml
```

O preset usa ASL-2nd com imputação e validação por sinalizador. O treino prepara etapas faltantes e reutiliza as concluídas. Para outro experimento:

```bash
libras-translator --train --config configs/landmark_lstm.toml
```

Entradas processadas ficam em `processed/`; resultados, em `runs/minds_libras/`. Veja opções e diferenças em relação ao artigo no [README principal](../../README.md).

Origem do download e licença: pendentes de documentação.
