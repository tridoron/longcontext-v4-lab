from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def without_paths(data: dict[str, Any], paths: set[tuple[str, ...]]) -> dict[str, Any]:
    copied = deepcopy(data)
    for path in paths:
        node = copied
        for key in path[:-1]:
            node = node[key]
        node.pop(path[-1], None)
    return copied


def test_attention_ablations_share_base_config_except_attention_type():
    baseline = load_yaml("configs/main_350m_hybrid.yaml")
    allowed = {
        ("model", "name"),
        ("model", "attention_type"),
        ("attention", "type"),
    }
    expected = without_paths(baseline, allowed)

    for path in [
        "configs/main_350m_full.yaml",
        "configs/main_350m_swa.yaml",
        "configs/main_350m_csa.yaml",
        "configs/main_350m_hca.yaml",
    ]:
        cfg = load_yaml(path)
        assert without_paths(cfg, allowed) == expected


def test_mhc_muon_differs_from_hybrid_only_in_mhc_and_muon_fields():
    baseline = load_yaml("configs/main_350m_hybrid.yaml")
    cfg = load_yaml("configs/main_350m_hybrid_mhc_muon.yaml")
    allowed = {
        ("model", "name"),
        ("mhc",),
        ("optimizer", "use_muon"),
    }

    assert without_paths(cfg, allowed) == without_paths(baseline, allowed)
    assert cfg["mhc"]["enabled"] is True
    assert cfg["optimizer"]["use_muon"] is True


def test_smoke_shards_are_independent_from_main_shards():
    main = load_yaml("configs/data_sources.yaml")
    smoke = load_yaml("configs/data_sources_smoke.yaml")

    for section in ["shards", "validation_shards"]:
        main_paths = {spec["path"] for spec in main[section].values()}
        smoke_paths = {spec["path"] for spec in smoke[section].values()}
        assert smoke_paths.isdisjoint(main_paths)
        assert all(path.startswith("data/smoke_shards/") for path in smoke_paths)
