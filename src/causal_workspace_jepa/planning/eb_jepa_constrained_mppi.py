"""Constraint-corrected MPPI compatible with the pinned EB-JEPA planner contract."""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import numpy as np
import torch


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor | None = None
    prev_elite_losses_mean: torch.Tensor | None = None
    prev_elite_losses_std: torch.Tensor | None = None
    info: dict | None = None


def project_action_norms(
    actions: torch.Tensor,
    *,
    max_norms: Optional[List[float]],
    max_norm_dims: Optional[List[int]],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Project selected action coordinates onto the configured L2 ball."""

    if max_norms is None:
        return actions
    if len(max_norms) != 1:
        raise ValueError("EB-JEPA Two Rooms supports exactly one action-norm bound")
    dimensions = (
        tuple(range(actions.shape[-1]))
        if max_norm_dims is None
        else tuple(int(dimension) for dimension in max_norm_dims)
    )
    if not dimensions or min(dimensions) < 0 or max(dimensions) >= actions.shape[-1]:
        raise ValueError("max_norm_dims must name valid action coordinates")
    selected = actions[..., list(dimensions)]
    norms = selected.norm(dim=-1, keepdim=True)
    coefficient = torch.clamp(float(max_norms[0]) / (norms + epsilon), max=1.0)
    projected = actions.clone()
    projected[..., list(dimensions)] = selected * coefficient
    return projected


class ConstrainedMPPIPlanner:
    """Pinned MPPI algorithm with candidate and returned-action norm projection."""

    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[int]] = None,
        decode_each_iteration: bool = False,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        del kwargs
        self.unroll = unroll
        self.objective: Callable | None = None
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel
        self._prev_mean = None

    def set_objective(self, objective: Callable) -> None:
        self.objective = objective

    def cost_function(self, actions: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if self.objective is None:
            raise RuntimeError("planner objective is not set")
        predicted_encodings = self.unroll(observation, actions)
        return self.objective(predicted_encodings)

    @torch.no_grad()
    def plan(
        self,
        obs_init,
        t0: bool = False,
        eval_mode: bool = False,
        steps_left=None,
        plan_vis_path=None,
    ) -> PlanningResult:
        del t0, plan_vis_path
        plan_length = self.plan_length if steps_left is None else min(self.plan_length, steps_left)
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )
        losses: list[float] = []
        elite_means: list[float] = []
        elite_stds: list[float] = []
        score = None
        elite_actions = None
        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )
            actions = project_action_norms(
                actions,
                max_norms=self.max_norms,
                max_norm_dims=self.max_norm_dims,
            )
            cost = self.cost_function(actions.permute(1, 2, 0), obs_init).unsqueeze(1)
            losses.append(cost.min().item())
            elite_indices = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_indices], actions[:, elite_indices]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))
            score /= score.sum(0)
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions,
                dim=1,
            ) / (score.sum(0) + 1e-9)
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
        if score is None or elite_actions is None:
            raise RuntimeError("n_iters must be positive")
        score_numpy = score.cpu().numpy()
        selected_actions = elite_actions[
            :, np.random.choice(np.arange(score_numpy.shape[0]), p=score_numpy)
        ]
        if not eval_mode:
            selected_actions += std * torch.randn_like(selected_actions)
        selected_actions = project_action_norms(
            selected_actions,
            max_norms=self.max_norms,
            max_norm_dims=self.max_norm_dims,
        )
        self._prev_mean = mean
        return PlanningResult(
            actions=selected_actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
