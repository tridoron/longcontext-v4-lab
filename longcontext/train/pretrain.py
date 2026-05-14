from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.dataset import PackedMemmapDataset
from longcontext.model.config import LongContextConfig
from longcontext.model.transformer import LongContextLM
from longcontext.optim.build_optimizer import build_optimizer
from longcontext.train.checkpoint import save_config
from longcontext.train.scheduler import build_cosine_scheduler
from longcontext.train.trainer import Trainer
from longcontext.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongContext 预训练入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只构建模型和优化器，不启动训练")
    parser.add_argument("--max-steps", type=int, default=None, help="显式限制训练步数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(raw.get("seed", 20260514)))
    config = LongContextConfig.from_dict(raw)
    model = LongContextLM(config)
    training_cfg = raw.get("training", {})
    if training_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    out_dir = Path(raw.get("output_dir", "outputs/artifacts")) / config.name
    optimizer = build_optimizer(model, raw.get("optimizer", {}), log_dir=out_dir)
    save_config(config, out_dir / "config.yaml")
    print(f"model={config.name} params={model.num_parameters():,}")
    if args.dry_run:
        print("dry-run: 已构建模型和优化器，未启动训练")
        return
    if args.max_steps is None:
        raise SystemExit("为避免误启动正式训练，必须显式传入 --max-steps。")
    data_cfg = raw["data"]
    dataset = PackedMemmapDataset(data_cfg["train_file"], seq_len=int(data_cfg["seq_len"]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_cfg = raw.get("optimizer", {})
    base_lr = float(optimizer_cfg.get("lr", 3.0e-4))
    min_lr = float(training_cfg.get("min_lr", optimizer_cfg.get("min_lr", base_lr * 0.1)))
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=int(training_cfg.get("warmup_steps", optimizer_cfg.get("warmup_steps", 2000))),
        total_steps=args.max_steps,
        min_lr_ratio=min_lr / base_lr,
    )
    trainer = Trainer(model, optimizer, loader, device, out_dir, scheduler=scheduler)
    trainer.train(max_steps=args.max_steps, save_every=int(raw.get("save_every", 1000)))


if __name__ == "__main__":
    main()
