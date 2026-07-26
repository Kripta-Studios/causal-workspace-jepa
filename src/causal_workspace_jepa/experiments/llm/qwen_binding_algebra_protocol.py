"""Pure preregistered protocol for causal binding-permutation algebra.

The protocol treats a permutation as an environment-like action over four
binding slots.  It deliberately contains no Qwen import, model forward,
activation capture, fitting, or scientific decision code.  Its purpose is to
freeze the finite action algebra and leakage-safe episode roster before any
protected model outcome exists.

Permutation convention
----------------------
A permutation maps each *source* slot to its destination slot.  Therefore,
``apply_permutation(values, p)[p[source]] == values[source]``.  The function
``compose_permutations(first, second)`` means "apply ``first`` and then
``second``", matching the temporal order used by a trajectory predictor.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np


SLOT_COUNT = 4
Permutation: TypeAlias = tuple[int, int, int, int]
PermutationClass: TypeAlias = str

IDENTITY_CLASS = "identity"
TRANSPOSITION_CLASS = "transposition"
DOUBLE_TRANSPOSITION_CLASS = "double_transposition"
THREE_CYCLE_CLASS = "three_cycle"
FOUR_CYCLE_CLASS = "four_cycle"
PERMUTATION_CLASSES = (
    IDENTITY_CLASS,
    TRANSPOSITION_CLASS,
    DOUBLE_TRANSPOSITION_CLASS,
    THREE_CYCLE_CLASS,
    FOUR_CYCLE_CLASS,
)
HELD_OUT_COMPOSITION_CLASSES = (
    DOUBLE_TRANSPOSITION_CLASS,
    THREE_CYCLE_CLASS,
    FOUR_CYCLE_CLASS,
)
IDENTITY_CONTROL = "identity_noop"
INVERSE_CONTROL = "inverse_restoration"
CONTROL_TYPES = (IDENTITY_CONTROL, INVERSE_CONTROL)

PRIMARY_TEMPLATE = (
    "Use the four mappings. Reply with only the value.\n"
    "- {k0} -> {v0}\n"
    "- {k1} -> {v1}\n"
    "- {k2} -> {v2}\n"
    "- {k3} -> {v3}\n"
    "Query: {query} ->"
)

PARAPHRASE_TEMPLATE = (
    "Read this lookup table and return only the requested entry.\n"
    "- {k0} = {v0}\n"
    "- {k1} = {v1}\n"
    "- {k2} = {v2}\n"
    "- {k3} = {v3}\n"
    "Value paired with {query}:"
)


@dataclass(frozen=True)
class BindingAlgebraEpisode:
    """One independent four-slot table before a permutation action."""

    episode_id: str
    split: str
    keys: tuple[str, str, str, str]
    base_values: tuple[str, str, str, str]
    query_index: int
    template: str = "primary"

    def __post_init__(self) -> None:
        if len(set(self.keys)) != SLOT_COUNT:
            raise ValueError("binding algebra episodes require four distinct keys")
        if len(set(self.base_values)) != SLOT_COUNT:
            raise ValueError("binding algebra episodes require four distinct values")
        if self.query_index not in range(SLOT_COUNT):
            raise ValueError("query_index must be in [0, 3]")
        if self.template not in {"primary", "paraphrase"}:
            raise ValueError(f"unknown binding template: {self.template}")

    @property
    def clean_answer(self) -> str:
        return self.base_values[self.query_index]

    def values_after(self, permutation: Sequence[int]) -> tuple[str, str, str, str]:
        """Return the table after the source-to-destination slot action."""

        values = apply_permutation(self.base_values, permutation)
        return values  # type: ignore[return-value]

    def answer_after(self, permutation: Sequence[int]) -> str:
        return self.values_after(permutation)[self.query_index]

    def prompt_after(self, permutation: Sequence[int]) -> str:
        return render_binding_algebra_prompt(
            self.keys,
            self.values_after(permutation),
            self.query_index,
            template=self.template,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BindingAlgebraCase:
    """One action rollout applied to an independent binding episode."""

    case_id: str
    split: str
    episode_id: str
    query_index: int
    target_permutation: Permutation
    permutation_class: PermutationClass
    generator_rollout: tuple[Permutation, ...]

    def __post_init__(self) -> None:
        target = validate_permutation(self.target_permutation)
        if self.permutation_class != permutation_class(target):
            raise ValueError("permutation_class does not match target_permutation")
        if not self.generator_rollout:
            raise ValueError("nonidentity algebra cases require a generator rollout")
        if any(permutation_class(action) != TRANSPOSITION_CLASS for action in self.generator_rollout):
            raise ValueError("generator_rollout must contain only transpositions")
        if compose_rollout(self.generator_rollout) != target:
            raise ValueError("generator_rollout does not compose to target_permutation")
        if self.query_index not in range(SLOT_COUNT):
            raise ValueError("query_index must be in [0, 3]")
        if not permutation_changes_slot(target, self.query_index):
            raise ValueError("registered cases must change the queried binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rollout_prefixes": rollout_prefixes(self.generator_rollout),
            "generator_action_matrices": predictor_generator_action_matrices(self),
        }


@dataclass(frozen=True)
class BindingAlgebraControlCase:
    """One explicitly registered identity or inverse-restoration control."""

    control_id: str
    split: str
    episode_id: str
    control_type: str
    parent_case_id: str | None
    action_rollout: tuple[Permutation, ...]
    expected_permutation: Permutation

    def __post_init__(self) -> None:
        if self.control_type not in CONTROL_TYPES:
            raise ValueError(f"unknown binding algebra control: {self.control_type}")
        if any(
            permutation_class(action) != TRANSPOSITION_CLASS
            for action in self.action_rollout
        ):
            raise ValueError("control rollouts must contain only transposition generators")
        expected = validate_permutation(self.expected_permutation)
        if compose_rollout(self.action_rollout) != expected:
            raise ValueError("control rollout does not compose to expected_permutation")
        if self.control_type == IDENTITY_CONTROL:
            if self.parent_case_id is not None or self.action_rollout:
                raise ValueError("identity controls must be empty episode-level no-ops")
        elif self.parent_case_id is None or expected != identity_permutation():
            raise ValueError("inverse controls must restore a named case to identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rollout_prefixes": rollout_prefixes(self.action_rollout),
        }


def validate_permutation(permutation: Sequence[int]) -> Permutation:
    """Validate and canonicalize one member of S4."""

    values = tuple(int(value) for value in permutation)
    if len(values) != SLOT_COUNT or set(values) != set(range(SLOT_COUNT)):
        raise ValueError("a binding permutation must contain each slot 0, 1, 2, 3 exactly once")
    return values  # type: ignore[return-value]


def identity_permutation() -> Permutation:
    return (0, 1, 2, 3)


def all_s4_permutations() -> tuple[Permutation, ...]:
    """Return all 24 actions in deterministic lexicographic order."""

    return tuple(
        validate_permutation(permutation)
        for permutation in itertools.permutations(range(SLOT_COUNT))
    )


def transposition(left: int, right: int) -> Permutation:
    """Return the action that swaps exactly two source/destination slots."""

    if left == right or left not in range(SLOT_COUNT) or right not in range(SLOT_COUNT):
        raise ValueError("a transposition requires two distinct slots in [0, 3]")
    result = list(identity_permutation())
    result[left], result[right] = result[right], result[left]
    return validate_permutation(result)


def transposition_generators() -> tuple[Permutation, ...]:
    """Return the six transpositions, the only train action class."""

    return tuple(
        transposition(left, right)
        for left in range(SLOT_COUNT)
        for right in range(left + 1, SLOT_COUNT)
    )


def apply_permutation(values: Sequence[Any], permutation: Sequence[int]) -> tuple[Any, ...]:
    """Move every value from its source slot to the registered destination."""

    if len(values) != SLOT_COUNT:
        raise ValueError("binding permutation actions require exactly four values")
    action = validate_permutation(permutation)
    result: list[Any] = [None] * SLOT_COUNT
    for source, destination in enumerate(action):
        result[destination] = values[source]
    return tuple(result)


def permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    """Materialize the source-to-destination action as a 4x4 matrix.

    For a column vector of slot values, ``apply_permutation`` is represented by
    ``matrix @ values``.  Temporal composition therefore satisfies
    ``M(first_then_second) == M(second) @ M(first)``.
    """

    action = validate_permutation(permutation)
    matrix = np.zeros((SLOT_COUNT, SLOT_COUNT), dtype=np.int64)
    for source, destination in enumerate(action):
        matrix[destination, source] = 1
    return matrix


def compose_permutations(
    first: Sequence[int], second: Sequence[int]
) -> Permutation:
    """Return the action produced by applying ``first`` and then ``second``."""

    left = validate_permutation(first)
    right = validate_permutation(second)
    return validate_permutation(tuple(right[left[source]] for source in range(SLOT_COUNT)))


def compose_rollout(actions: Sequence[Sequence[int]]) -> Permutation:
    """Compose a temporally ordered action sequence from left to right."""

    result = identity_permutation()
    for action in actions:
        result = compose_permutations(result, action)
    return result


def rollout_prefixes(
    actions: Sequence[Sequence[int]],
) -> tuple[Permutation, ...]:
    """Return identity followed by every cumulative temporal prefix."""

    prefixes = [identity_permutation()]
    for action in actions:
        prefixes.append(compose_permutations(prefixes[-1], action))
    return tuple(prefixes)


def inverse_permutation(permutation: Sequence[int]) -> Permutation:
    """Return the unique action that restores every source slot."""

    action = validate_permutation(permutation)
    inverse = [0] * SLOT_COUNT
    for source, destination in enumerate(action):
        inverse[destination] = source
    return validate_permutation(inverse)


def permutation_cycle_lengths(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return a canonical descending cycle-length signature."""

    action = validate_permutation(permutation)
    visited: set[int] = set()
    lengths: list[int] = []
    for start in range(SLOT_COUNT):
        if start in visited:
            continue
        cursor = start
        length = 0
        while cursor not in visited:
            visited.add(cursor)
            cursor = action[cursor]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def permutation_class(permutation: Sequence[int]) -> PermutationClass:
    """Classify an S4 action by its conjugacy/cycle class."""

    signature = permutation_cycle_lengths(permutation)
    classes = {
        (1, 1, 1, 1): IDENTITY_CLASS,
        (2, 1, 1): TRANSPOSITION_CLASS,
        (2, 2): DOUBLE_TRANSPOSITION_CLASS,
        (3, 1): THREE_CYCLE_CLASS,
        (4,): FOUR_CYCLE_CLASS,
    }
    return classes[signature]


def permutations_in_classes(classes: Sequence[str]) -> tuple[Permutation, ...]:
    """Select a deterministic action roster and reject unknown classes."""

    requested = tuple(str(value) for value in classes)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("permutation classes must be nonempty and unique")
    unknown = set(requested).difference(PERMUTATION_CLASSES)
    if unknown:
        raise ValueError(f"unknown permutation classes: {sorted(unknown)}")
    return tuple(
        action for action in all_s4_permutations() if permutation_class(action) in requested
    )


def permutation_changes_slot(permutation: Sequence[int], slot: int) -> bool:
    """Whether an action changes the value occupying ``slot``."""

    action = validate_permutation(permutation)
    if slot not in range(SLOT_COUNT):
        raise ValueError("slot must be in [0, 3]")
    return action[slot] != slot


def decompose_into_transpositions(permutation: Sequence[int]) -> tuple[Permutation, ...]:
    """Return a deterministic minimal rollout of train-class generators."""

    action = validate_permutation(permutation)
    visited: set[int] = set()
    rollout: list[Permutation] = []
    for start in range(SLOT_COUNT):
        if start in visited:
            continue
        cycle: list[int] = []
        cursor = start
        while cursor not in visited:
            visited.add(cursor)
            cycle.append(cursor)
            cursor = action[cursor]
        for member in cycle[1:]:
            rollout.append(transposition(cycle[0], member))
    result = tuple(rollout)
    if compose_rollout(result) != action:
        raise RuntimeError("internal transposition decomposition error")
    return result


def predictor_generator_action_matrices(
    case: BindingAlgebraCase,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Encode only the primitive program, never a direct composed target."""

    if any(
        permutation_class(action) != TRANSPOSITION_CLASS
        for action in case.generator_rollout
    ):
        raise ValueError("predictor action programs may contain only train generators")
    return tuple(
        tuple(tuple(int(value) for value in row) for row in permutation_matrix(action))
        for action in case.generator_rollout
    )


def render_binding_algebra_prompt(
    keys: Sequence[str],
    values: Sequence[str],
    query_index: int,
    *,
    template: str,
) -> str:
    """Render one fixed four-slot lookup prompt."""

    if len(keys) != SLOT_COUNT or len(values) != SLOT_COUNT:
        raise ValueError("the binding algebra benchmark is fixed to exactly four pairs")
    if query_index not in range(SLOT_COUNT):
        raise ValueError("query_index must be in [0, 3]")
    if template not in {"primary", "paraphrase"}:
        raise ValueError(f"unknown template: {template}")
    pattern = PRIMARY_TEMPLATE if template == "primary" else PARAPHRASE_TEMPLATE
    fields = {f"k{index}": key for index, key in enumerate(keys)}
    fields.update({f"v{index}": value for index, value in enumerate(values)})
    fields["query"] = keys[query_index]
    return pattern.format(**fields)


def generate_binding_algebra_episodes(
    *,
    split: str,
    keys: Sequence[str],
    values: Sequence[str],
    count: int,
    seed: int,
    template: str = "primary",
) -> list[BindingAlgebraEpisode]:
    """Generate deterministic, balanced, duplicate-free base episodes."""

    if len(keys) < SLOT_COUNT or len(values) < SLOT_COUNT:
        raise ValueError("each split requires at least four keys and values")
    if len(set(keys)) != len(keys) or len(set(values)) != len(values):
        raise ValueError("split pools must not contain duplicates")
    if count <= 0:
        raise ValueError("episode count must be positive")
    rng = np.random.default_rng(seed)
    episodes: list[BindingAlgebraEpisode] = []
    signatures: set[tuple[Any, ...]] = set()
    attempts = 0
    while len(episodes) < count and attempts < count * 100:
        attempts += 1
        index = len(episodes)
        episode_keys = tuple(str(value) for value in rng.choice(keys, size=4, replace=False))
        episode_values = tuple(
            str(value) for value in rng.choice(values, size=4, replace=False)
        )
        query_index = index % SLOT_COUNT
        signature = (episode_keys, episode_values, query_index, template)
        if signature in signatures:
            continue
        signatures.add(signature)
        episodes.append(
            BindingAlgebraEpisode(
                episode_id=f"{split}-{index:04d}",
                split=split,
                keys=episode_keys,  # type: ignore[arg-type]
                base_values=episode_values,  # type: ignore[arg-type]
                query_index=query_index,
                template=template,
            )
        )
    if len(episodes) != count:
        raise RuntimeError("could not generate the requested unique binding algebra episodes")
    return episodes


def assert_globally_disjoint_token_pools(
    token_pools: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    """Reject token reuse across every role and independent split."""

    owners: dict[str, tuple[str, str]] = {}
    for role, pools in token_pools.items():
        for split, values in pools.items():
            for raw_value in values:
                value = str(raw_value)
                if value in owners:
                    previous = owners[value]
                    raise ValueError(
                        f"token {value!r} appears in {previous!r} and {(role, split)!r}"
                    )
                owners[value] = (str(role), str(split))


def assert_action_class_partition(
    train_classes: Sequence[str], held_out_classes: Sequence[str]
) -> None:
    """Reject overlap between primitive train actions and composed evaluation actions."""

    train = tuple(str(value) for value in train_classes)
    held_out = tuple(str(value) for value in held_out_classes)
    permutations_in_classes(train)
    permutations_in_classes(held_out)
    overlap = set(train).intersection(held_out)
    if overlap:
        raise ValueError(f"action classes leak across train and held-out: {sorted(overlap)}")
    if train != (TRANSPOSITION_CLASS,):
        raise ValueError("the preregistered train action class must be transposition only")
    if set(held_out) != set(HELD_OUT_COMPOSITION_CLASSES):
        raise ValueError("held-out actions must be the three nonidentity composition classes")


def binding_algebra_episodes_from_config(
    config: Mapping[str, Any],
) -> list[BindingAlgebraEpisode]:
    """Materialize independent splits plus an exact paired paraphrase shift."""

    pools = config["token_pools"]
    assert_globally_disjoint_token_pools(pools)
    episodes: list[BindingAlgebraEpisode] = []
    by_split: dict[str, list[BindingAlgebraEpisode]] = {}
    for split in ("calibration", "train", "validation", "test", "paraphrase"):
        split_config = config["splits"][split]
        paired_with = split_config.get("paired_with")
        if paired_with is not None:
            source = by_split.get(str(paired_with))
            if source is None:
                raise ValueError(f"paired source split {paired_with!r} is not available")
            if int(split_config["count"]) != len(source):
                raise ValueError("paired split count must equal its source split count")
            split_episodes = [
                BindingAlgebraEpisode(
                    episode_id=f"{split}-{index:04d}",
                    split=split,
                    keys=episode.keys,
                    base_values=episode.base_values,
                    query_index=episode.query_index,
                    template=str(split_config["template"]),
                )
                for index, episode in enumerate(source)
            ]
        else:
            split_episodes = generate_binding_algebra_episodes(
                split=split,
                keys=pools["keys"][split],
                values=pools["values"][split],
                count=int(split_config["count"]),
                seed=int(split_config["seed"]),
                template=str(split_config["template"]),
            )
        by_split[split] = split_episodes
        episodes.extend(split_episodes)
    return episodes


def binding_algebra_cases_from_config(
    config: Mapping[str, Any],
) -> list[BindingAlgebraCase]:
    """Expand base episodes into primitive-train and composed-evaluation cases."""

    partition = config["action_partition"]
    train_classes = tuple(str(value) for value in partition["train_classes"])
    held_out_classes = tuple(str(value) for value in partition["held_out_classes"])
    assert_action_class_partition(train_classes, held_out_classes)
    episodes = binding_algebra_episodes_from_config(config)
    cases: list[BindingAlgebraCase] = []
    split_counts: dict[str, int] = {}
    for episode in episodes:
        classes = (
            train_classes
            if episode.split in {"calibration", "train"}
            else held_out_classes
        )
        roster = tuple(
            action
            for action in permutations_in_classes(classes)
            if permutation_changes_slot(action, episode.query_index)
        )
        for action in roster:
            index = split_counts.get(episode.split, 0)
            split_counts[episode.split] = index + 1
            cases.append(
                BindingAlgebraCase(
                    case_id=f"{episode.split}-case-{index:05d}",
                    split=episode.split,
                    episode_id=episode.episode_id,
                    query_index=episode.query_index,
                    target_permutation=action,
                    permutation_class=permutation_class(action),
                    generator_rollout=decompose_into_transpositions(action),
                )
            )
    return cases


def binding_algebra_controls(
    episodes: Sequence[BindingAlgebraEpisode],
    cases: Sequence[BindingAlgebraCase],
) -> list[BindingAlgebraControlCase]:
    """Materialize identity and inverse controls into the protocol roster."""

    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("binding algebra episode IDs must be unique")
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    controls = [
        BindingAlgebraControlCase(
            control_id=f"{episode.episode_id}:identity",
            split=episode.split,
            episode_id=episode.episode_id,
            control_type=IDENTITY_CONTROL,
            parent_case_id=None,
            action_rollout=(),
            expected_permutation=identity_permutation(),
        )
        for episode in episodes
    ]
    for case in cases:
        episode = episode_by_id.get(case.episode_id)
        if episode is None or episode.split != case.split:
            raise ValueError("binding algebra case refers to an unknown or wrong-split episode")
        inverse_rollout = decompose_into_transpositions(
            inverse_permutation(case.target_permutation)
        )
        controls.append(
            BindingAlgebraControlCase(
                control_id=f"{case.case_id}:inverse",
                split=case.split,
                episode_id=case.episode_id,
                control_type=INVERSE_CONTROL,
                parent_case_id=case.case_id,
                action_rollout=case.generator_rollout + inverse_rollout,
                expected_permutation=identity_permutation(),
            )
        )
    if len({control.control_id for control in controls}) != len(controls):
        raise RuntimeError("binding algebra control IDs are not unique")
    return controls


def phase_access_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the preregistered phase/split storage policy."""

    policy = config["phase_policy"]
    access = policy["access"]
    observed = {
        phase: tuple(str(split) for split in access[phase]["allowed_splits"])
        for phase in ("phase_0", "phase_1_train", "protected_eval")
    }
    expected = {
        "phase_0": ("calibration", "train", "validation"),
        "phase_1_train": ("train", "validation"),
        "protected_eval": ("test", "paraphrase"),
    }
    if observed != expected:
        raise ValueError("phase access differs from the preregistered split policy")
    roots = {
        phase: str(access[phase]["output_root"])
        for phase in ("phase_0", "phase_1_train", "protected_eval")
    }
    if len(set(roots.values())) != len(roots):
        raise ValueError("phase outputs must use three distinct roots")
    if str(policy["phase_0_decision_split"]) != "validation":
        raise ValueError("phase 0 decisions must use validation only")
    if policy.get("protected_requires_frozen_phase_1") is not True:
        raise ValueError("protected access must require a frozen phase-1 checkpoint and plan")
    return {
        "allowed_splits": observed,
        "output_roots": roots,
        "phase_0_decision_split": "validation",
        "protected_requires_frozen_phase_1": True,
    }


def assert_phase_split_access(
    config: Mapping[str, Any],
    phase: str,
    requested_splits: Sequence[str],
) -> tuple[str, ...]:
    """Fail closed when a runner asks a phase to open an unregistered split."""

    contract = phase_access_contract(config)
    if phase not in contract["allowed_splits"]:
        raise ValueError(f"unknown binding algebra phase: {phase}")
    requested = tuple(str(split) for split in requested_splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested splits must be nonempty and unique")
    allowed = set(contract["allowed_splits"][phase])
    forbidden = set(requested).difference(allowed)
    if forbidden:
        raise RuntimeError(
            f"BLOCKED_SPLIT_ACCESS: {phase} cannot open {sorted(forbidden)}"
        )
    return requested


def binding_algebra_protocol_digest(
    config: Mapping[str, Any],
    episodes: Sequence[BindingAlgebraEpisode],
    cases: Sequence[BindingAlgebraCase],
    controls: Sequence[BindingAlgebraControlCase],
) -> str:
    """Hash the ordered semantic roster without using model or tokenizer state."""

    phase_contract = phase_access_contract(config)
    protocol_contract = {
        "permutation_convention": config["permutation_convention"],
        "action_partition": config["action_partition"],
        "treatment": config["treatment"],
        "capture": config["capture"],
        "meta_model_action_contract": {
            key: config["meta_model"][key]
            for key in (
                "action_encoding",
                "train_rollout_length",
                "evaluation_rollout_lengths",
                "token_ids_as_features_forbidden",
                "train_on_composed_targets_forbidden",
                "composed_target_matrix_as_input_forbidden",
            )
        },
        "baselines": config["baselines"],
        "controls": config["controls"],
        "phase_access": phase_contract,
    }
    payload = {
        "episodes": [episode.to_dict() for episode in episodes],
        "cases": [case.to_dict() for case in cases],
        "control_cases": [control.to_dict() for control in controls],
        "protocol_contract": protocol_contract,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
