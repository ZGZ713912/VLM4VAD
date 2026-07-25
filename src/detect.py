from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from detection.scorer import VideoAnomalyDetector


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def _load_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}

    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM4VAD batch inference")
    parser.add_argument("--config")
    parser.add_argument("--video-dir")
    parser.add_argument("--anomaly-text")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--window-stride", type=int)
    parser.add_argument("--window-topk-ratio", type=float)
    parser.add_argument("--video-topk", type=int)
    parser.add_argument("--smoothing-kernel", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--output")
    parser.set_defaults(recursive=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    root = _repo_root()

    video_dir = args.video_dir or config.get("video_dir")
    anomaly_text = args.anomaly_text or config.get("anomaly_text")
    checkpoint = args.checkpoint or config.get("model_checkpoint")
    if not video_dir or not anomaly_text or not checkpoint:
        raise SystemExit("--video-dir, --anomaly-text, and --checkpoint are required")

    visual_length = config.get("visual_length", 256)
    window_stride = args.window_stride if args.window_stride is not None else config.get("window_stride")
    if window_stride is None:
        window_stride = max(1, visual_length // 2)

    detector = VideoAnomalyDetector(
        checkpoint_path=_resolve_path(checkpoint, root),
        device=args.device or config.get("device"),
        threshold=args.threshold if args.threshold is not None else config.get("threshold", 0.5),
        frame_stride=args.frame_stride if args.frame_stride is not None else config.get("frame_stride", 16),
        max_frames=args.max_frames if args.max_frames is not None else config.get("max_frames"),
        visual_length=visual_length,
        visual_width=config.get("visual_width", 512),
        embed_dim=config.get("embed_dim", 512),
        visual_head=config.get("visual_head", 1),
        visual_layers=config.get("visual_layers", 1),
        attn_window=config.get("attn_window", 64),
        prompt_prefix=config.get("prompt_prefix", 10),
        prompt_postfix=config.get("prompt_postfix", 10),
        window_stride=window_stride,
        smoothing_kernel=args.smoothing_kernel if args.smoothing_kernel is not None else config.get("smoothing_kernel", 5),
        window_topk_ratio=args.window_topk_ratio if args.window_topk_ratio is not None else config.get("window_topk_ratio", 0.125),
        video_topk=args.video_topk if args.video_topk is not None else config.get("video_topk", 2),
    )

    recursive = config.get("recursive", True) if args.recursive is None else args.recursive
    results = detector.predict_folder(_resolve_path(video_dir, root), anomaly_text, recursive=recursive)
    payload = [result.to_dict() for result in results]

    for result in results:
        state = "abnormal" if result.abnormal else "normal"
        print(f"{Path(result.video_path).name}\t{state}\t{result.score:.4f}\t{result.canonical_anomaly_text}")

    output = args.output or config.get("output")
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = (root / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
