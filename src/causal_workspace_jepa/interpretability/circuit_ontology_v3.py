"""Circuit-ontology v3: preserve registered outcomes while separating functional notions.

The ontology is intentionally descriptive.  It never upgrades or rewrites the registered
scientific disposition of an experiment.  Its purpose is to keep distinct questions that are
frequently conflated in mechanistic-interpretability work:

* full graph recovery,
* epsilon-functional sufficiency,
* necessity,
* redundancy-group coverage,
* cancellation-group coverage, and
* equivalence-class identification.

Only measurements that were actually executed prospectively may be marked measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


NOT_MEASURED = "NOT_MEASURED_PROSPECTIVELY"
MEASURED = "MEASURED"


@dataclass(frozen=True)
class FunctionalSufficiency:
    """Faithfulness of a frozen selected mechanism on an exact confirmation split."""

    recovery_fraction: float
    epsilon: float
    threshold: float
    passes_threshold: bool


@dataclass(frozen=True)
class GraphRecovery:
    """Literal overlap with the planted/registered graph, kept separate from sufficiency."""

    node_precision: float | None
    node_recall: float | None
    edge_precision: float | None
    edge_recall: float | None


@dataclass(frozen=True)
class ProspectiveMeasurement:
    """A measurement whose availability is explicit instead of silently inferred."""

    status: str
    value: Any = None
    note: str = ""


def epsilon_functional_sufficiency(
    recovery_fraction: float,
    *,
    threshold: float = 0.95,
) -> FunctionalSufficiency:
    """Return a descriptive epsilon-sufficiency score.

    ``epsilon`` is the unexplained fraction under the benchmark's own recovery definition.
    This function does not decide the experiment's registered status.
    """

    recovery = float(recovery_fraction)
    return FunctionalSufficiency(
        recovery_fraction=recovery,
        epsilon=max(0.0, 1.0 - recovery),
        threshold=float(threshold),
        passes_threshold=bool(recovery >= threshold),
    )


def graph_recovery_from_confirmation(payload: Mapping[str, Any]) -> GraphRecovery:
    """Extract literal node/edge overlap metrics when the benchmark exposes them."""

    def _optional_float(key: str) -> float | None:
        value = payload.get(key)
        return None if value is None else float(value)

    return GraphRecovery(
        node_precision=_optional_float("node_precision"),
        node_recall=_optional_float("node_recall"),
        edge_precision=_optional_float("edge_precision"),
        edge_recall=_optional_float("edge_recall"),
    )


def registered_gate_failures(gates: Mapping[str, Any]) -> list[str]:
    """Return exactly the registered boolean gates that failed."""

    return sorted(key for key, value in gates.items() if value is False)


def group_coverage(
    selected: Iterable[str],
    groups: Mapping[str, Sequence[str]],
    *,
    any_member_suffices: bool,
) -> dict[str, dict[str, Any]]:
    """Score explicit groups without pretending they were preregistered if they were not.

    This primitive is used by future prospective benchmarks.  Retrospective HARD-002 audit code
    records its redundancy/cancellation groups as ``NOT_MEASURED_PROSPECTIVELY`` rather than using
    this helper to rescue the historical node-recall gate.
    """

    chosen = set(selected)
    result: dict[str, dict[str, Any]] = {}
    for name, members in groups.items():
        member_set = set(members)
        overlap = sorted(chosen & member_set)
        covered = bool(overlap) if any_member_suffices else member_set.issubset(chosen)
        result[name] = {
            "members": list(members),
            "selected_members": overlap,
            "covered": covered,
            "rule": "any_member" if any_member_suffices else "all_members",
        }
    return result


def conservative_v3_record(
    *,
    registered_status: str,
    gates: Mapping[str, Any],
    iid_confirmation: Mapping[str, Any],
    ood_confirmation: Mapping[str, Any],
    functional_threshold: float = 0.95,
    prospective_necessity: ProspectiveMeasurement | None = None,
    prospective_redundancy: ProspectiveMeasurement | None = None,
    prospective_cancellation: ProspectiveMeasurement | None = None,
    prospective_equivalence: ProspectiveMeasurement | None = None,
) -> dict[str, Any]:
    """Build the ontology record while preserving the original disposition verbatim."""

    necessity = prospective_necessity or ProspectiveMeasurement(
        NOT_MEASURED,
        note="No prospectively frozen necessity intervention was executed for this ontology.",
    )
    redundancy = prospective_redundancy or ProspectiveMeasurement(
        NOT_MEASURED,
        note="Redundancy-group coverage was not a registered acceptance rule for this run.",
    )
    cancellation = prospective_cancellation or ProspectiveMeasurement(
        NOT_MEASURED,
        note="Cancellation-group coverage was not a registered acceptance rule for this run.",
    )
    equivalence = prospective_equivalence or ProspectiveMeasurement(
        NOT_MEASURED,
        note="Circuit equivalence classes were not prospectively enumerated or confirmed.",
    )

    return {
        "registered_status_preserved": str(registered_status),
        "registered_gate_failures": registered_gate_failures(gates),
        "iid": {
            "functional_sufficiency": asdict(
                epsilon_functional_sufficiency(
                    float(iid_confirmation["circuit_recovery_fraction"]),
                    threshold=functional_threshold,
                )
            ),
            "full_graph_recovery": asdict(graph_recovery_from_confirmation(iid_confirmation)),
        },
        "ood": {
            "functional_sufficiency": asdict(
                epsilon_functional_sufficiency(
                    float(ood_confirmation["circuit_recovery_fraction"]),
                    threshold=functional_threshold,
                )
            ),
            "full_graph_recovery": asdict(graph_recovery_from_confirmation(ood_confirmation)),
        },
        "necessity": asdict(necessity),
        "redundancy_group_coverage": asdict(redundancy),
        "cancellation_group_coverage": asdict(cancellation),
        "equivalence_class_identification": asdict(equivalence),
        "adjudication_rule": (
            "Descriptive ontology only. The registered_status_preserved field is authoritative; "
            "functional sufficiency cannot retroactively convert a negative registered result."
        ),
    }
