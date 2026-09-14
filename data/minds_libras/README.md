# MINDS-Libras

Coloque os vídeos extraídos diretamente em `data/minds_libras/raw/`, sem renomeá-los.

Subconjunto utilizado: 800 vídeos, 20 classes, 8 sinalizadores e 5 repetições.

Exemplo: `01AcontecerSinalizador05-3.mp4` corresponde à classe 01 (Acontecer),
sinalizador 05 e repetição 3.

Na raiz do repositório, gere o manifesto:

```bash
libras-translator --build-manifest data/minds_libras
```

Saída: `data/minds_libras/metadata/manifest.csv`.

Origem do download e licença: pendentes de documentação.
