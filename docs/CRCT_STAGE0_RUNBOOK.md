# CRCT Stage-0 Circuit-Recovery Benchmark

Status: `IMPLEMENTED_UNEXECUTED_ON_TARGET_HOST`.

Experiment: `CRCT-STAGE0-001`.

Base commit for the patch: `f69cc28f00faf9d5382e3a47a551410785ae9374`.

## Purpose

This benchmark is the next falsification layer after the existing Causal-Residual Stage-0 study.
It does **not** ask only whether a learner predicts a finite residual. It constructs a synthetic
system with a known sparse causal circuit and asks whether the analysis recovers the planted
residual-producing nodes while rejecting differential-only and high-variance nuisance components.

The benchmark separates

```text
finite intervention effect
    = first/second-order differential transport
    + finite nonlinear residual.
```

The synthetic plant contains:

- an explicit linear path;
- an explicit quadratic path;
- a sparse sigmoid/tanh multiplicative routing path;
- a sparse sinusoidal bypass path;
- disconnected high-variance nuisance coordinates;
- inactive route/bypass coordinates that look like ordinary activations but have zero readout;
- a function-preserving diagonal gauge transform that strongly rescales internal coordinates.

The ground-truth residual circuit is known before evaluation. Therefore node ranking can be scored
with average precision and precision/recall at the true circuit size rather than by visual inspection.

## Required scientific behavior

A passing synthetic run must establish all of the following on every registered seed:

1. the finite residual retains at least five percent of full effect energy;
2. second-order Taylor transport beats first order;
3. direct residual-causal ranking recovers the planted sparse circuit with AP >= 0.90 and precision@k >= 0.80;
4. the selected top-k residual circuit beats the p95 of equal-cardinality random node sets;
5. a function-preserving diagonal gauge changes coordinate magnitudes without changing the model output or residual-causal score;
6. a train-only residual MLP improves validation and held-out test residual prediction by at least 50 percent;
7. adding the learned residual back to the second-order transport reaches full-effect held-out replay NMSE <= 0.20.

A pass supports only this statement:

> The CRCT implementation distinguishes differential effects, finite nonlinear residuals, nuisance
> magnitude, and sparse causal residual nodes in a controlled synthetic system.

It does **not** establish a Qwen circuit, a JEPA mechanism, a global workspace, or a SOTA result.

## Profiles

`smoke` is a small CPU/GPU validation profile. `full` is the primary RTX-class profile and is the
default in the PowerShell wrapper. `max` increases sample count, student width/depth, optimization
steps, diagnostic rows, and matched random controls. The registered primary decision should use
`full`; `max` is a scaling/robustness follow-up and must not replace a failed primary after results
are observed.

The default registered seeds are `7, 13, 23`. A separate process-level smoke preflight uses seed `20260817`. By default it uses `auto`, so CUDA-capable scientific runs validate the actual accelerator/runtime path; the targeted pytest suite already exercises the CPU path. The preflight has a hard 300 s timeout so a native backend hang cannot silently block diagnostics.

## Outputs

Every suite run creates an ignored directory under
`artifacts/reports/crct_stage0/CRCT-STAGE0-001_<timestamp>/` containing:

- `config.snapshot.json` — exact suite config used;
- `base_guard.json` — repository revision check;
- `diagnostics/environment.json` — Python/Torch/CUDA/system metadata;
- `diagnostics/git_diff.txt` and Git status/log/diff-check snapshots;
- `diagnostics/nvidia_smi_*.txt`, `nvcc_version.txt`, and package inventory;
- `logs/pytest.txt`;
- `logs/preflight.txt`;
- one stdout/stderr log per scientific seed;
- `metrics/preflight.json`;
- one complete JSON result per registered seed;
- `aggregate.json`;
- `metrics.csv`;
- `summary.md`;
- `SUITE_STATUS.json`;
- `MANIFEST.json` containing SHA-256 hashes for every retained diagnostic file.

The runner also creates a sibling `.zip` automatically. That ZIP is the preferred artifact to bring
back for external review because it contains raw per-seed metrics and diagnostics, not only a summary.

## Target-host command

From repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')"

pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_crct_stage0.ps1 `
  -Profile full `
  -Device auto `
  -Seeds 7,13,23
```

The wrapper defaults the process-level smoke preflight to `auto` with a 300 s hard timeout.
To make this explicit on Windows/SM120:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_crct_stage0.ps1 `
  -Profile full `
  -Device auto `
  -Seeds 7,13,23 `
  -PreflightDevice auto `
  -PreflightTimeoutSeconds 300
```

`-SkipPreflight` is only for a deliberate diagnostic rerun after a separately recorded smoke
preflight. It must not be used to bypass a failing scientific gate.

Do not silently use a CPU Torch build for `full`/`max`. The suite fails closed unless CUDA is visible,
unless the explicit `-AllowCpuFull` override is supplied.

For an additional scaling run **after preserving the full-profile result**:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_crct_stage0.ps1 `
  -Profile max `
  -Device cuda `
  -Seeds 7,13,23,37,101
```

Do not change thresholds after seeing a result. If a gate fails, retain the ZIP and diagnose the
failure before changing code or declaring a discovery.
