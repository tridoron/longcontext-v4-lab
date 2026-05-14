from __future__ import annotations

from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from longcontext.model.config import LongContextConfig
from longcontext.model.transformer import LongContextLM


def load_config_and_model(
    config_path: str | Path,
    weights_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[LongContextConfig, LongContextLM, torch.device]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = LongContextConfig.from_dict(raw)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LongContextLM(config)
    if weights_path is not None:
        state = load_file(str(weights_path))
        model.load_state_dict(state, strict=True)
    model.to(dev)
    model.eval()
    return config, model, dev


def write_csv(path: str | Path, rows: list[dict]) -> None:
    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
