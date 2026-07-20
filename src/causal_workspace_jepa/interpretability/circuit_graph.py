"""Circuit graph schema and JSON/GraphML I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CircuitNode:
    node_id: str
    site: str
    evidence_level: str
    score: float


@dataclass(frozen=True)
class CircuitEdge:
    source: str
    target: str
    effect: float
    faithfulness: float


@dataclass(frozen=True)
class CircuitGraph:
    graph_id: str
    nodes: tuple[CircuitNode, ...]
    edges: tuple[CircuitEdge, ...]
    status: str = "SCAFFOLDED"

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "status": self.status,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CircuitGraph":
        return cls(
            graph_id=str(payload["graph_id"]),
            status=str(payload.get("status", "SCAFFOLDED")),
            nodes=tuple(CircuitNode(**node) for node in payload.get("nodes", [])),  # type: ignore[arg-type]
            edges=tuple(CircuitEdge(**edge) for edge in payload.get("edges", [])),  # type: ignore[arg-type]
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "CircuitGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_graphml(self, path: str | Path) -> None:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            f'  <graph id="{self.graph_id}" edgedefault="directed">',
        ]
        for node in self.nodes:
            lines.append(f'    <node id="{node.node_id}" />')
        for index, edge in enumerate(self.edges):
            lines.append(
                f'    <edge id="e{index}" source="{edge.source}" target="{edge.target}" />'
            )
        lines.extend(["  </graph>", "</graphml>"])
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
