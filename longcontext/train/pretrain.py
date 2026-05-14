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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongContext 预训练入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只构建模型和优化器，不启动训练")
    parser.add_argument("--max-steps", type=int, default=None, help="显式限制训练步数")
    return parser.parse_args()


def resolve_project_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def resolve_shard_path(data_cfg: dict, key_name: str, fallback_name: str | None = None) -> Path:
    sources_path = resolve_project_path(data_cfg.get("sources_config", "configs/data_sources.yaml"))
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    shard_key = data_cfg.get(key_name)
    if shard_key:
        shard_spec = sources.get("shards", {}).get(shard_key)
        if shard_spec is None:
            raise KeyError(f"{sources_path} 未定义 shards.{shard_key}")
        shard_path = shard_spec["path"] if isinstance(shard_spec, dict) else shard_spec
        return resolve_project_path(shard_path)
    if fallback_name is not None and data_cfg.get(fallback_name):
        return resolve_project_path(data_cfg[fallback_name])
    raise KeyError(f"data 配置必须提供 {key_name}，并在 {sources_path} 的 shards 中声明路径")


def require_cuda_device(amp_dtype: str = "bf16") -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，训练已退出；请检查驱动、PyTorch CUDA 版本和 CUDA_VISIBLE_DEVICES。")
    device = torch.device("cuda")
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        device_name = torch.cuda.get_device_name(device)
        raise SystemExit(f"当前 CUDA 设备/驱动不支持 bf16 autocast，训练已退出: {device_name}")
    return device


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config, Path.cwd())
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    train_file = resolve_shard_path(data_cfg, "train_key", "train_file")
    micro_batch_size = int(data_cfg.get("micro_batch_size", data_cfg.get("batch_size", 1)))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    amp_dtype = str(training_cfg.get("amp_dtype", "bf16"))
    seq_len = int(data_cfg["seq_len"])
    configured_tokens_per_step = training_cfg.get("tokens_per_step")
    actual_tokens_per_step = micro_batch_size * grad_accum_steps * seq_len
    if configured_tokens_per_step is not None and int(configured_tokens_per_step) != actual_tokens_per_step:
        raise SystemExit(
            "tokens_per_step 配置不一致: "
            f"configured={configured_tokens_per_step}, actual={actual_tokens_per_step}"
        )
    print(
        "effective_batch="
        f"{micro_batch_size} micro_batch × {grad_accum_steps} grad_accum × {seq_len} seq_len "
        f"= {actual_tokens_per_step:,} tokens/step"
    )
    dataset = PackedMemmapDataset(train_file, seq_len=seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
    )
    device = require_cuda_device(amp_dtype)
    optimizer_cfg = raw.get("optimizer", {})
    base_lr = float(optimizer_cfg.get("lr", 3.0e-4))
    min_lr = float(training_cfg.get("min_lr", optimizer_cfg.get("min_lr", base_lr * 0.1)))
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=int(training_cfg.get("warmup_steps", optimizer_cfg.get("warmup_steps", 2000))),
        total_steps=args.max_steps,
        min_lr_ratio=min_lr / base_lr,
    )
    trainer = Trainer(
        model,
        optimizer,
        loader,
        device,
        out_dir,
        scheduler=scheduler,
        amp_dtype=amp_dtype,
        grad_accum_steps=grad_accum_steps,
        milestone_tokens=training_cfg.get("milestone_tokens", []),
        target_seen_tokens=training_cfg.get("seen_tokens"),
        strip_optimizer_state_on_finish=bool(training_cfg.get("strip_optimizer_state_on_finish", False)),
    )
    trainer.train(max_steps=args.max_steps, save_every=int(raw.get("save_every", 1000)))


if __name__ == "__main__":
    main()
