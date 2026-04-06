"""
FusionOps Environment
OpenEnv-compatible RL environment for ML graph scheduling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Action, Config, Graph, ScheduleState, SubgraphEntry,
)
from .cost_model import compute_subgraph_latency, compute_naive_latency
from .validator import validate_action
from .observation import format_observation


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class FusionOpsEnv:
    """
    RL environment for ML graph scheduling.
    The agent builds an execution schedule step by step.
    """

    def __init__(
        self,
        graph: Graph,
        max_steps: int = 20,
        allow_recomputation: bool = True,
    ):
        self.graph = graph
        self.max_steps = max_steps
        self.allow_recomputation = allow_recomputation

        # Precompute naive baseline for grading
        self.naive_latency = compute_naive_latency(graph)

        self.state: Optional[ScheduleState] = None
        self._last_action_result = ""

    def reset(self) -> StepResult:
        """Initialize the environment. Returns initial observation."""
        self.state = ScheduleState()
        self._last_action_result = ""

        obs = format_observation(
            graph=self.graph,
            state=self.state,
            last_action_result="",
            last_action_error=None,
            max_steps=self.max_steps,
            naive_latency=self.naive_latency,
        )

        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"naive_latency": self.naive_latency},
        )

    def _classify_error(self, error_msg: str) -> dict:
        """Classify an error message into type + fix hint for the LLM."""
        msg = (error_msg or "").lower()
        ready_ops = self._compute_ready_ops()

        if "oom" in msg or "working set" in msg or "exceeds" in msg:
            return {
                "type": "Memory Error (working set too large)",
                "reason": error_msg,
                "fix_hint": "Reduce tile size (e.g., config=[64,64,1]) or use split-K (smaller k for MatMul). Working set must fit in fast memory.",
            }
        if "dependency" in msg or "needs tensor" in msg or "not scheduled" in msg:
            return {
                "type": "Dependency Error (op not ready)",
                "reason": error_msg,
                "fix_hint": f"Choose ops only from READY OPS: {ready_ops}",
            }
        if "retain" in msg and "not produced" in msg:
            return {
                "type": "Retention Error (tensor not produced by current subgraph)",
                "reason": error_msg,
                "fix_hint": "You can only retain tensors that the CURRENT subgraph produces (its outputs).",
            }
        if "connected" in msg or "subgraph" in msg:
            return {
                "type": "Connectivity Error (ops do not form connected subgraph)",
                "reason": error_msg,
                "fix_hint": "Ops in a subgraph must form a connected DAG (one op produces a tensor consumed by another in the same subgraph).",
            }
        if "parse" in msg or "format" in msg:
            return {
                "type": "Parse Error (invalid action format)",
                "reason": error_msg,
                "fix_hint": "Use exact format: SCHEDULE ops=[0,1] config=[128,128,1] retain=[]",
            }
        if "already scheduled" in msg or "duplicate" in msg:
            return {
                "type": "Already Scheduled (op already executed)",
                "reason": error_msg,
                "fix_hint": f"This op was already executed. Choose from READY OPS: {ready_ops}",
            }
        # Default
        return {
            "type": "Invalid Action",
            "reason": error_msg,
            "fix_hint": f"Choose ops from READY OPS: {ready_ops}. Use format: SCHEDULE ops=[op_id] config=[128,128,1] retain=[]",
        }

    def _compute_ready_ops(self) -> list:
        """Find ops whose predecessors are all scheduled."""
        if self.state is None:
            return []
        ready = []
        for op in self.graph.operations:
            if op.id in self.state.scheduled_op_ids:
                continue
            preds = self.graph.op_predecessors.get(op.id, set())
            if preds.issubset(self.state.scheduled_op_ids):
                ready.append(op.id)
        return ready

    def _penalty_for_error(self, error_type: str) -> float:
        """Graduated penalties based on error type."""
        # Match by prefix to handle the descriptive tags
        if error_type.startswith("Parse Error"):
            return -0.10
        if error_type.startswith("Memory Error"):
            return -0.05  # Close to valid, just wrong size
        if error_type.startswith("Dependency Error"):
            return -0.20  # Clearly wrong order
        if error_type.startswith("Retention Error"):
            return -0.15
        if error_type.startswith("Connectivity Error"):
            return -0.15
        if error_type.startswith("Already Scheduled"):
            return -0.10
        return -0.10

    def step(self, action: Action) -> StepResult:
        """Execute one scheduling step."""
        assert self.state is not None, "Must call reset() before step()"

        # Handle recomputation: temporarily allow re-scheduling of ops
        state_for_validation = self.state
        if self.allow_recomputation:
            state_for_validation = self.state.clone()
            for oid in action.operation_ids:
                state_for_validation.scheduled_op_ids.discard(oid)

        # Validate
        validation = validate_action(self.graph, action, state_for_validation)
        if not validation.is_valid:
            self.state.step_count += 1
            error_info = self._classify_error(validation.error)
            penalty = self._penalty_for_error(error_info["type"])

            done = self.state.step_count >= self.max_steps
            obs = format_observation(
                graph=self.graph,
                state=self.state,
                last_action_result="",
                last_action_error=error_info,
                max_steps=self.max_steps,
                naive_latency=self.naive_latency,
            )

            return StepResult(
                observation=obs,
                reward=penalty,
                done=done,
                info={"error": validation.error, "error_type": error_info["type"]},
            )

        # Compute latency
        result = compute_subgraph_latency(
            self.graph, action, state_for_validation
        )

        if not result.is_valid:
            self.state.step_count += 1
            error_info = self._classify_error(result.error)
            penalty = self._penalty_for_error(error_info["type"])

            done = self.state.step_count >= self.max_steps
            obs = format_observation(
                graph=self.graph,
                state=self.state,
                last_action_result="",
                last_action_error=error_info,
                max_steps=self.max_steps,
                naive_latency=self.naive_latency,
            )

            return StepResult(
                observation=obs,
                reward=penalty,
                done=done,
                info={"error": result.error, "error_type": error_info["type"]},
            )

        # Update state
        self._update_state(action, result.total_latency)

        # Check if done
        all_ops_covered = self._all_ops_covered()
        step_limit = self.state.step_count >= self.max_steps
        done = all_ops_covered or step_limit

        # Compute reward
        reward = self._compute_reward(result.total_latency, all_ops_covered, done)

        last_result_str = (
            f"VALID. Latency={result.total_latency:.1f}, "
            f"working_set={result.working_set:.0f}"
        )

        obs = format_observation(
            graph=self.graph,
            state=self.state,
            last_action_result=last_result_str,
            last_action_error=None,
            max_steps=self.max_steps,
            naive_latency=self.naive_latency,
        )

        info = {
            "latency": result.total_latency,
            "working_set": result.working_set,
            "tile_latencies": result.tile_latencies,
            "total_latency": self.state.total_latency,
            "all_ops_covered": all_ops_covered,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def get_state(self) -> dict:
        """Return full environment state for serialization."""
        assert self.state is not None
        return {
            "scheduled_op_ids": list(self.state.scheduled_op_ids),
            "tensors_in_fast_memory": list(self.state.tensors_in_fast_memory),
            "total_latency": self.state.total_latency,
            "step_count": self.state.step_count,
            "naive_latency": self.naive_latency,
            "schedule": [
                {
                    "ops": e.operation_ids,
                    "config": [e.config.w, e.config.h, e.config.k],
                    "retain": e.tensors_to_retain,
                    "latency": e.latency,
                }
                for e in self.state.schedule_history
            ],
        }

    def get_score(self) -> float:
        """
        Compute final score in [0, 1].
        Score = naive_latency / (naive_latency + agent_latency)
        This gives:
          - 0.5 if agent matches naive (no improvement)
          - approaching 1.0 as agent latency approaches 0
          - below 0.5 if agent is worse than naive
        Then we remap [0.5, 1.0] -> [0.0, 1.0] so that:
          - matching naive = 0.0
          - optimal (much better than naive) = approaching 1.0
        """
        if self.state is None or self.state.total_latency <= 0:
            return 0.0
        if not self._all_ops_covered():
            # Penalty: not all ops covered
            covered = len(self._covered_ops())
            total = len(self.graph.operations)
            coverage_ratio = covered / total if total > 0 else 0
            return coverage_ratio * 0.1  # max 0.1 if incomplete

        ratio = self.naive_latency / (self.naive_latency + self.state.total_latency)
        # ratio is in (0, 1). naive match gives 0.5. Better gives > 0.5.
        # Remap: score = (ratio - 0.5) * 2, clamped to [0, 1]
        # But also handle case where agent is worse than naive (ratio < 0.5)
        score = max(0.0, (ratio - 0.5) * 2.0)
        return min(score, 1.0)

    def _update_state(self, action: Action, latency: float):
        """Update schedule state after a valid action."""
        # Mark ops as scheduled
        for oid in action.operation_ids:
            self.state.scheduled_op_ids.add(oid)

        # Update fast memory:
        # 1. Remove all previously retained tensors that aren't in the new retain list
        #    (they get evicted when a new subgraph runs)
        # Actually, the spec says tensors_to_retain controls what stays AFTER this subgraph.
        # Everything else is evicted. So we clear fast memory and only keep retained tensors.

        # Clear fast memory
        self.state.tensors_in_fast_memory.clear()

        # Add retained tensors
        for tid in action.tensors_to_retain:
            self.state.tensors_in_fast_memory.add(tid)

        # Record in history
        entry = SubgraphEntry(
            operation_ids=list(action.operation_ids),
            config=action.config,
            tensors_to_retain=list(action.tensors_to_retain),
            traversal_order=action.traversal_order,
            latency=latency,
        )
        self.state.schedule_history.append(entry)

        # Update counters
        self.state.total_latency += latency
        self.state.step_count += 1

    def _covered_ops(self) -> set[int]:
        """All ops that have been scheduled at least once."""
        return set(self.state.scheduled_op_ids)

    def _all_ops_covered(self) -> bool:
        """Check if every operation has been scheduled at least once."""
        all_op_ids = set(op.id for op in self.graph.operations)
        return all_op_ids.issubset(self._covered_ops())

    def _compute_reward(
        self, step_latency: float, all_ops_covered: bool, done: bool
    ) -> float:
        """
        Compute per-step reward. Higher is better.
        Combines efficiency signal with completion incentive.
        """
        # Small positive baseline for any valid step (encourages exploration)
        valid_step_bonus = 0.02

        # Base reward: negative normalized latency (lower latency = higher reward)
        avg_naive_per_op = self.naive_latency / len(self.graph.operations)
        latency_reward = -step_latency / avg_naive_per_op * 0.1

        # Bonus for fusion (multiple ops in one subgraph)
        num_ops_in_step = len(
            self.state.schedule_history[-1].operation_ids
        ) if self.state.schedule_history else 1
        fusion_bonus = (num_ops_in_step - 1) * 0.05

        # Completion bonus
        completion_bonus = 0.0
        if done and all_ops_covered:
            score = self.naive_latency / self.state.total_latency
            completion_bonus = score * 0.5

        return valid_step_bonus + latency_reward + fusion_bonus + completion_bonus


def parse_action(text: str, graph: Graph) -> Optional[Action]:
    """
    Parse an action from LLM text output.
    Expected format: SCHEDULE ops=[0,1] config=[128,128,1] retain=[2]
    Flexible parsing to handle variations.
    """
    text = text.strip()

    # Extract operation IDs
    ops_match = re.search(r'ops\s*=\s*\[([^\]]*)\]', text)
    if not ops_match:
        # Try simpler format: just numbers
        ops_match = re.search(r'ops\s*=\s*(\d[\d,\s]*)', text)
    if not ops_match:
        return None

    try:
        ops_str = ops_match.group(1)
        op_ids = [int(x.strip()) for x in ops_str.split(',') if x.strip()]
    except (ValueError, IndexError):
        return None

    # Extract config [w, h, k]
    config_match = re.search(r'config\s*=\s*\[([^\]]*)\]', text)
    if not config_match:
        # Default to native granularity
        nw, nh = graph.hardware.native_granularity
        config = Config(nw, nh, 1)
    else:
        try:
            parts = [int(x.strip()) for x in config_match.group(1).split(',')]
            if len(parts) == 3:
                config = Config(parts[0], parts[1], parts[2])
            elif len(parts) == 2:
                config = Config(parts[0], parts[1], 1)
            else:
                nw, nh = graph.hardware.native_granularity
                config = Config(nw, nh, 1)
        except (ValueError, IndexError):
            nw, nh = graph.hardware.native_granularity
            config = Config(nw, nh, 1)

    # Extract retain list
    retain_match = re.search(r'retain\s*=\s*\[([^\]]*)\]', text)
    if retain_match:
        try:
            retain_str = retain_match.group(1).strip()
            if retain_str:
                retain = [int(x.strip()) for x in retain_str.split(',') if x.strip()]
            else:
                retain = []
        except (ValueError, IndexError):
            retain = []
    else:
        retain = []

    # Extract traversal order (optional)
    trav_match = re.search(r'traversal\s*=\s*\[([^\]]*)\]', text)
    traversal = None
    if trav_match:
        try:
            trav_str = trav_match.group(1).strip()
            if trav_str:
                traversal = [int(x.strip()) for x in trav_str.split(',')]
        except (ValueError, IndexError):
            traversal = None

    return Action(
        operation_ids=op_ids,
        config=config,
        tensors_to_retain=retain,
        traversal_order=traversal,
    )
