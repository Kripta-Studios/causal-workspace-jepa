"""Local finite-difference Jacobian-lens approximation.

Experiment modules build prompt-local and corpus-averaged transports on this
primitive.  The official Anthropic package is registered as an external
reference and is not vendored; the implemented baselines are independently
tested approximations, not a claim of package equivalence.
"""

from causal_workspace_jepa.hooks.gradients import finite_difference_jacobian

__all__ = ["finite_difference_jacobian"]
