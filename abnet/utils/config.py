"""OmegaConf-based configuration with YAML inheritance and CLI overrides."""

import os
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_root() -> str:
    return _REPO_ROOT


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def _load_with_base(path: str, _seen=None) -> DictConfig:
    """Load a YAML file, recursively merging anything listed under ``_base_``."""
    path = _resolve(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ reference at {path}")
    _seen.add(path)

    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg

    if isinstance(bases, str):
        bases = [bases]

    merged = OmegaConf.create()
    for base in bases:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        if not os.path.exists(base_path):
            base_path = _resolve(base)
        merged = OmegaConf.merge(merged, _load_with_base(base_path, _seen))
    return OmegaConf.merge(merged, cfg)


def load_config(cfg_path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load ``cfg_path`` and apply ``key=value`` dotlist overrides."""
    cfg = _load_with_base(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def config_to_str(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg)
