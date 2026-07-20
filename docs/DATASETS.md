# Datasets

Status: `SCAFFOLDED`.

Tier 0 datasets are generated locally from explicit seeds and must remain below 512 MB by default. Tier 1/Tier 2 datasets and model weights are not downloaded on the CPU VPS.

## Tier 0 Mandatory

| Name | Source | License | Modalities | Shapes | Splits | OOD | Resource Modes | Purpose | Leakage Risks | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PointMass2D | local generator | MIT code, generated data | state/action | state 4, action 2 | trajectory seed split | mass, drag, force, noise | `cpu_vps` | action-conditioned dynamics | trajectory overlap | `NOT_STARTED` |
| BouncingBall2D | local generator | MIT code, generated data | state/action | state 4, action 1 | trajectory seed split | gravity, restitution, obstacles | `cpu_vps` | contact-free physics | seed overlap | `NOT_STARTED` |
| TwoBodyCollision | local generator | MIT code, generated data | state/action/contact | state TBD | trajectory seed split | mass swap, removal, restitution | `cpu_vps` | interaction causality | donor/recipient overlap | `NOT_STARTED` |
| MiniPush | local generator | MIT code, generated data | pixels/state/masks/action | 32x32 default | layout/object split | object/layout shift | `cpu_vps` smoke | object-local interventions | mask leakage into eval | `NOT_STARTED` |
| TinyMaze | local generator | MIT code, generated data | grid/state/action/value | TBD | layout/goal split | layout/task shift | `cpu_vps` | planning/workspace tests | shared layouts | `NOT_STARTED` |

## Tier 1 And Tier 2

LeWorldModel, JEPA-WMs, C-JEPA, V-JEPA, DROID, RoboCasa, ManiSkill, and SkyJEPA assets are `BLOCKED_RESOURCE` or `BLOCKED_EXTERNAL` on this VPS.
