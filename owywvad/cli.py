from __future__ import annotations

import argparse
from typing import Sequence

from owywvad.config import load_config
from owywvad.data.cache import build_cache
from owywvad.data.fetch import fetch_dataset
from owywvad.data.prepare import prepare_dataset
from owywvad.deps import install_dependencies
from owywvad.eval.evaluate import evaluate_dataset
from owywvad.infer.runner import infer_video
from owywvad.reproduce.paper import reproduce_paper_main
from owywvad.train.runner import train_stage


def _print_messages(payload: list[str] | dict[str, object]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    for item in payload:
        print(item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m owywvad", description="OW-YW-VAD reference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    deps_parser = sub.add_parser("deps")
    deps_sub = deps_parser.add_subparsers(dest="deps_cmd", required=True)
    deps_sub.add_parser("install")

    data_parser = sub.add_parser("data")
    data_sub = data_parser.add_subparsers(dest="data_cmd", required=True)
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("dataset")
    fetch.add_argument("--source", default=None)
    fetch.add_argument("--config", default="configs/paper_main.yaml")
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("dataset")
    prepare.add_argument("--config", default="configs/paper_main.yaml")

    cache_parser = sub.add_parser("cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_cmd", required=True)
    build = cache_sub.add_parser("build")
    build.add_argument("dataset")
    build.add_argument("--config", default="configs/paper_main.yaml")

    train = sub.add_parser("train")
    train.add_argument("stage", choices=["stage2", "stage3", "stage4"])
    train.add_argument("--config", default="configs/paper_main.yaml")

    infer_parser = sub.add_parser("infer")
    infer_sub = infer_parser.add_subparsers(dest="infer_cmd", required=True)
    video = infer_sub.add_parser("video")
    video.add_argument("--input", required=True)
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--config", default="configs/paper_main.yaml")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--config", default="configs/paper_main.yaml")

    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("recipe", choices=["paper-main"])
    reproduce.add_argument("--config", default="configs/paper_main.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "deps":
        _print_messages(install_dependencies())
        return 0
    if args.command == "data":
        config = load_config(args.config)
        if args.data_cmd == "fetch":
            _print_messages(fetch_dataset(args.dataset, config, args.source))
        else:
            _print_messages(prepare_dataset(args.dataset, config))
        return 0
    if args.command == "cache":
        _print_messages(build_cache(args.dataset, load_config(args.config)))
        return 0
    if args.command == "train":
        _print_messages(train_stage(args.stage, load_config(args.config)))
        return 0
    if args.command == "infer":
        _print_messages(infer_video(args.input, args.checkpoint, load_config(args.config)))
        return 0
    if args.command == "evaluate":
        _print_messages(evaluate_dataset(args.dataset, args.checkpoint, load_config(args.config)))
        return 0
    if args.command == "reproduce":
        _print_messages(reproduce_paper_main(load_config(args.config)))
        return 0
    return 1
