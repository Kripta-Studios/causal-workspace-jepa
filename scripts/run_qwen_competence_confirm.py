from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/qwen_competence_confirm_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from causal_workspace_jepa.experiments.llm.qwen_binding_competence_confirm import (
        main as run_main,
    )

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = ROOT / "artifacts/reports/qwen_competence_confirm" / config["experiment_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    config["execution"]["device"] = args.device
    tmp_config = run_dir / "config.runtime.json"
    tmp_config.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return run_main(
        [
            "--config",
            str(tmp_config),
            "--ledger",
            str(run_dir / "ACCESS_LEDGER.jsonl"),
            "--output",
            str(run_dir / "metrics.json"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
