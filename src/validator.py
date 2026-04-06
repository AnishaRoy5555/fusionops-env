"""
FusionOps Validator
Validates agent actions before cost computation.
Checks: op validity, dependency satisfaction, subgraph connectivity, memory limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Action, Graph, OpType, ScheduleState
from .cost_model import classify_tensors, compute_tile_geometry, compute_working_set


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


def validate_action(
    graph: Graph,
    action: Action,
    state: ScheduleState,
) -> ValidationResult:
    """
    Validate an agent's action before execution.
    Returns ValidationResult with error message if invalid.
    """
    op_ids = action.operation_ids
    config = action.config

    # 1. Check operation IDs are valid
    valid_op_ids = set(range(len(graph.operations)))
    for oid in op_ids:
        if oid not in valid_op_ids:
            return ValidationResult(False, f"Invalid operation ID: {oid}")

    # 2. Check operations are not already scheduled
    for oid in op_ids:
        if oid in state.scheduled_op_ids:
            return ValidationResult(False, f"Operation {oid} already scheduled")

    # 3. Check no duplicates in action
    if len(set(op_ids)) != len(op_ids):
        return ValidationResult(False, "Duplicate operation IDs in action")

    # 4. Check dependencies are satisfied
    # Every input tensor to the subgraph that comes from another op
    # must have its producer already scheduled (or be produced within this subgraph)
    op_id_set = set(op_ids)
    for oid in op_ids:
        op = graph.get_op(oid)
        for tid in op.input_tensor_ids:
            if tid in graph.tensor_producer:
                producer_id = graph.tensor_producer[tid]
                if producer_id not in state.scheduled_op_ids and producer_id not in op_id_set:
                    return ValidationResult(
                        False,
                        f"Dependency not met: op {oid} needs tensor {tid} "
                        f"from op {producer_id} which is not scheduled"
                    )

    # 5. Check subgraph connectivity
    # The ops must form a connected subgraph in the DAG
    if len(op_ids) > 1:
        if not _is_connected(graph, op_ids):
            return ValidationResult(False, "Operations do not form a connected subgraph")

    # 6. Check config validity
    if config.w <= 0 or config.h <= 0 or config.k <= 0:
        return ValidationResult(False, f"Config dimensions must be positive: [{config.w},{config.h},{config.k}]")

    # Check config divides tensor dimensions evenly or tiles correctly
    # (w and h just need to be <= tensor dims, tiling handles the rest)
    # But they should be powers of 2 or at least reasonable
    # For now, just check they're positive (the cost model handles tiling)

    # 7. Check tensors_to_retain validity
    # Can only retain tensors that are outputs of this subgraph
    produced_by_subgraph = set()
    for oid in op_ids:
        op = graph.get_op(oid)
        produced_by_subgraph.update(op.output_tensor_ids)

    for tid in action.tensors_to_retain:
        if tid not in produced_by_subgraph:
            return ValidationResult(
                False,
                f"Cannot retain tensor {tid}: not produced by this subgraph"
            )

    # 8. Check working set fits in fast memory
    tensor_class = classify_tensors(graph, op_ids, state)
    ws = compute_working_set(graph, tensor_class, config, op_ids, state)
    if ws > graph.hardware.fast_memory_capacity:
        return ValidationResult(
            False,
            f"OOM: working set {ws:.0f} exceeds fast memory capacity "
            f"{graph.hardware.fast_memory_capacity}"
        )

    # 9. Check traversal order validity (if provided)
    if action.traversal_order is not None:
        geom = compute_tile_geometry(graph, op_ids, config)
        expected_tiles = geom.total_spatial_tiles
        if len(action.traversal_order) != expected_tiles:
            return ValidationResult(
                False,
                f"Traversal order length {len(action.traversal_order)} "
                f"!= expected tiles {expected_tiles}"
            )
        if set(action.traversal_order) != set(range(expected_tiles)):
            return ValidationResult(
                False,
                "Traversal order must be a permutation of tile indices"
            )

    return ValidationResult(True)


def _is_connected(graph: Graph, op_ids: list[int]) -> bool:
    """
    Check if the given ops form a connected subgraph.
    Connected means: for any two ops in the set, there exists a path
    between them through tensor edges (ignoring direction).
    """
    op_id_set = set(op_ids)
    if len(op_id_set) <= 1:
        return True

    # Build undirected adjacency within the subgraph
    adj: dict[int, set[int]] = {oid: set() for oid in op_ids}

    for oid in op_ids:
        op = graph.get_op(oid)
        # Check if any output tensor of this op is consumed by another op in the subgraph
        for tid in op.output_tensor_ids:
            if tid in graph.tensor_consumers:
                for consumer_id in graph.tensor_consumers[tid]:
                    if consumer_id in op_id_set and consumer_id != oid:
                        adj[oid].add(consumer_id)
                        adj[consumer_id].add(oid)

        # Check if any input tensor of this op is produced by another op in the subgraph
        for tid in op.input_tensor_ids:
            if tid in graph.tensor_producer:
                producer_id = graph.tensor_producer[tid]
                if producer_id in op_id_set and producer_id != oid:
                    adj[oid].add(producer_id)
                    adj[producer_id].add(oid)

    # BFS from first op
    visited = set()
    queue = [op_ids[0]]
    visited.add(op_ids[0])

    while queue:
        current = queue.pop(0)
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited == op_id_set
