from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class TrainConfig:
    target_model: str = "gpt2-large"
    draft_model: str = "gpt2"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    lr: float = 5e-5
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 2_000
    warmup_steps: int = 100
    max_seq_len: int = 512
    seed: int = 0

    # Left None so harness.plan_device decides from the GPU we land on.
    dtype: str | None = None
    out_dir: str = "/kaggle/working/draft-ckpt"
    log_every: int = 50
    tags: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum
