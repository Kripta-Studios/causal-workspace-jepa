from causal_workspace_jepa.experiments.world_model.eb_jepa_training_resources import (
    evaluate_training_resource_profile,
)


REVISION = "966e61e9285b3a876f49b9774e9720d9a99a7925"


def _row(batch_size: int, mode: str, status: str = "SUCCESS") -> dict:
    return {
        "source": {"revision": REVISION, "clean": True},
        "runtime": {
            "torch": "2.10.0+cu128",
            "capability": [12, 0],
            "arch_list": ["sm_120"],
        },
        "profile": {"batch_size": batch_size, "mode": mode},
        "training": {
            "status": status,
            "losses_finite": status == "SUCCESS",
            "gradients_finite": status == "SUCCESS",
            "parameter_delta": 0.001 if status == "SUCCESS" else 0.0,
            "peak_reserved_bytes": 1234 if status == "SUCCESS" else 0,
        },
    }


def test_resource_profile_accepts_compile_failure_as_diagnostic_data() -> None:
    rows = [_row(64, "eager"), _row(384, "eager"), _row(64, "compile", "FAILED")]
    passes = evaluate_training_resource_profile(
        rows, revision=REVISION, minimum_eager_batch=64
    )
    assert all(passes.values())


def test_resource_profile_requires_minimum_eager_training() -> None:
    rows = [_row(64, "eager", "FAILED"), _row(64, "compile", "FAILED")]
    passes = evaluate_training_resource_profile(
        rows, revision=REVISION, minimum_eager_batch=64
    )
    assert not passes["minimum_eager_batch_trains"]
