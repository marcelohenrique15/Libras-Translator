# MINDS-Libras

Coloque os vídeos em `data/minds_libras/raw/`, diretamente ou em subpastas por
sinalizador, sem renomeá-los.

O manifesto atual tem 1.155 vídeos de 20 classes e 12 sinalizadores. A divisão
considera apenas as repetições de 1 a 5.

Exemplo: `01AcontecerSinalizador05-3.mp4` corresponde à classe 01 (Acontecer),
sinalizador 05 e repetição 3.

Na raiz do repositório, gere o manifesto:

```bash
libras-translator --build-manifest data/minds_libras --test-signer-id 05
```

Saída: `data/minds_libras/metadata/manifest.csv`, com a coluna `set` indicando
`train` ou `test` conforme o sinalizador escolhido.

Origem do download e licença: pendentes de documentação.
