"""Read-only ontology-v3 audit of the retained CRCT HARD-002 bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any

from causal_workspace_jepa.interpretability.circuit_ontology_v3 import conservative_v3_record

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_ROOT = ROOT / "artifacts/reports/crct_stage0_hard"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_hard002_zip(report_root: Path = DEFAULT_REPORT_ROOT) -> Path | None:
    """Return the most recent retained HARD-002 bundle without mutating it."""

    if not report_root.exists():
        return None
    matches = sorted(
        report_root.glob("CRCT-STAGE0-HARD-002_*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def audit_hard002_zip(path: Path, *, functional_threshold: float = 0.95) -> dict[str, Any]:
    """Re-express HARD-002 under ontology v3 while preserving its registered status."""

    path = Path(path)
    if not path.exists():
        return {
            "schema_version": "circuit_ontology_v3_hard002_audit_v1",
            "status": "MISSING_HARD002_BUNDLE",
            "path": str(path),
        }

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        suite = json.loads(archive.read("SUITE_STATUS.json"))
        aggregate = json.loads(archive.read("aggregate.json"))
        manifest = json.loads(archive.read("MANIFEST.json"))
        seed_paths = sorted(
            name
            for name in names
            if name.startswith("metrics/seed_") and name.endswith(".json")
        )
        seeds: list[dict[str, Any]] = []
        for seed_path in seed_paths:
            payload = json.loads(archive.read(seed_path))
            ontology = conservative_v3_record(
                registered_status=str(payload["status"]),
                gates=payload["gates"],
                iid_confirmation=payload["iid_confirmation"],
                ood_confirmation=payload["ood_confirmation"],
                functional_threshold=functional_threshold,
            )
            seeds.append(
                {
                    "seed": int(payload["seed"]),
                    "result_sha256": payload.get("result_sha256"),
                    "selected": payload["frozen_discovery_plan"]["selected"],
                    "truth_nodes": payload["ground_truth"].get("truth_nodes", []),
                    "truth_edges": payload["ground_truth"].get("truth_edges", []),
                    "residual_power_fraction": payload["differential_diagnostics"][
                        "iid_residual_power_fraction"
                    ],
                    "ontology_v3": ontology,
                }
            )

    output = {
        "schema_version": "circuit_ontology_v3_hard002_audit_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "READ_ONLY_AUDIT_COMPLETE",
        "source_bundle": {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        },
        "registered_suite_status_preserved": suite.get("status"),
        "registered_aggregate_status_preserved": aggregate.get("status"),
        "manifest_schema": manifest.get("schema_version"),
        "functional_threshold": functional_threshold,
        "seeds": seeds,
        "interpretation": {
            "historical_disposition_changed": False,
            "node_recall_retuned": False,
            "new_confirmatory_claim_created": False,
            "note": (
                "HARD-002 remains governed by its original gates. Ontology v3 only separates "
                "literal graph recall from epsilon-functional sufficiency. Necessity, redundancy-"
                "group coverage, cancellation-group coverage, and circuit equivalence were not "
                "prospectively measured in HARD-002 and remain NOT_MEASURED_PROSPECTIVELY."
            ),
        },
    }
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-zip", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional-threshold", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = args.hard_zip or find_latest_hard002_zip()
    if path is None:
        payload = {
            "schema_version": "circuit_ontology_v3_hard002_audit_v1",
            "status": "MISSING_HARD002_BUNDLE",
            "report_root": str(DEFAULT_REPORT_ROOT),
        }
    else:
        payload = audit_hard002_zip(path, functional_threshold=args.functional_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
