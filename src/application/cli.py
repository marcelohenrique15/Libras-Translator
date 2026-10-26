"""Entrada da aplicação; treinamento e câmera continuam independentes."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from application.config import RuntimeConfig, resolve_checkpoint


def main():
    parser = argparse.ArgumentParser(prog="libras-camera", description="Reconhecimento de sinais isolados em câmera ou vídeo.")
    parser.add_argument("--config", type=Path, help="Configuração TOML da aplicação")
    parser.add_argument("--checkpoint", type=Path, help="Modelo treinado; padrão: execução concluída mais recente em runs/")
    parser.add_argument("--camera", type=int, help="Índice da câmera, por exemplo 0 ou 1")
    parser.add_argument("--video", type=Path, help="Arquivo local para demonstração/reprodução")
    parser.add_argument("--device", help="auto, cpu ou cuda:0")
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--headless", action="store_true", help="Executa sem abrir janela")
    parser.add_argument("--max-seconds", type=float, help="Limita a duração da sessão, útil para verificação")
    args = parser.parse_args()
    if args.camera is not None and args.video is not None:
        parser.error("Escolha --camera ou --video, uma única fonte.")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("--max-seconds deve ser positivo.")
    if args.video is not None and not args.video.is_file():
        parser.error(f"Vídeo não encontrado: {args.video}")
    try:
        config = RuntimeConfig.from_toml(args.config) if args.config else RuntimeConfig()
        source = str(args.video.resolve()) if args.video is not None else args.camera
        config = resolve_checkpoint(config.with_overrides(
            checkpoint=args.checkpoint, source=source, device=args.device,
            window_seconds=args.window_seconds, confidence_threshold=args.confidence_threshold,
            mirror=args.mirror,
        ))
        from application.runtime.app import CameraApplication
        print(f"Modelo: {config.checkpoint}\nFonte: {config.source}\nESC sair · ESPAÇO pausar · R limpar", flush=True)
        stats = CameraApplication(config, headless=args.headless, max_seconds=args.max_seconds).run()
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Erro: {error}\n")
    except KeyboardInterrupt:
        print("\nSessão encerrada.")
