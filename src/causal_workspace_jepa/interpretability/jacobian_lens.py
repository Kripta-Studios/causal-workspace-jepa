"""Jacobian-lens approximations.

This module provides CPU finite-difference scaffolding only. Official
Anthropic Jacobian Lens integration is `BLOCKED_RESOURCE` for this VPS.
"""

from causal_workspace_jepa.hooks.gradients import finite_difference_jacobian

__all__ = ["finite_difference_jacobian"]
