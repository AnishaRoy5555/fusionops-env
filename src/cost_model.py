"""
FusionOps Cost Model
Computes subgraph execution latency using the roofline model.
Handles tiling, split-K, data reuse, and memory transfers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .models import (
    Action, Config, Graph, OpType, ScheduleState, Tensor, TensorRole,
)


@dataclass
class TensorClassification:
    """How each tensor relates to a subgraph."""
    boundary_inputs: dict[int, str]   # tensor_id -> role_in_op ("pointwise", "lhs", "rhs")
    boundary_outputs: set[int]        # tensor_ids that leave the subgraph
    ephemeral: set[int]               # tensor_ids internal to subgraph
    resident_inputs: set[int]         # tensor_ids already in fast memory


@dataclass
class TileGeometry:
    """Tiling structure for a subgraph execution."""
    num_tiles_w: int
    num_tiles_h: int
    total_spatial_tiles: int
    num_k_steps: int
    total_iterations: int
    output_w: int  # actual output tensor width
    output_h: int  # actual output tensor height
    reduction_dim: int  # K for MatMul, 0 for pure Pointwise


@dataclass
class SubgraphLatencyResult:
    """Full result of latency computation."""
    total_latency: float
    tile_latencies: list[float]
    working_set: float
    is_valid: bool
    error: Optional[str] = None


def classify_tensors(
    graph: Graph,
    op_ids: list[int],
    state: ScheduleState,
) -> TensorClassification:
    """Classify every tensor touched by this subgraph."""
    op_id_set = set(op_ids)

    # All tensors produced within this subgraph
    produced_internally = set()
    for oid in op_ids:
        op = graph.get_op(oid)
        produced_internally.update(op.output_tensor_ids)

    # All tensors consumed within this subgraph
    consumed_internally = set()
    for oid in op_ids:
        op = graph.get_op(oid)
        consumed_internally.update(op.input_tensor_ids)

    # Boundary inputs: consumed but not produced internally
    boundary_input_ids = consumed_internally - produced_internally

    # Classify boundary inputs by their role
    boundary_inputs: dict[int, str] = {}
    for oid in op_ids:
        op = graph.get_op(oid)
        for tid in op.input_tensor_ids:
            if tid in boundary_input_ids:
                if op.op_type == OpType.MATMUL:
                    # First input is LHS, second is RHS
                    idx = op.input_tensor_ids.index(tid)
                    role = "lhs" if idx == 0 else "rhs"
                else:
                    role = "pointwise"
                boundary_inputs[tid] = role

    # Separate resident inputs (already in fast memory)
    resident_inputs = set()
    non_resident_inputs: dict[int, str] = {}
    for tid, role in boundary_inputs.items():
        if tid in state.tensors_in_fast_memory:
            resident_inputs.add(tid)
        else:
            non_resident_inputs[tid] = role

    # Classify produced tensors as ephemeral or boundary output.
    # A tensor is ephemeral if it is produced AND consumed within the subgraph
    # (even if it also has external consumers - those are served by recomputation).
    # A tensor is a boundary output if it is produced by the subgraph and:
    #   - NOT consumed by any op within the subgraph, OR
    #   - Is a graph output (must eventually reach slow memory)
    # But for cost purposes, even graph outputs that are retained don't get evicted now.
    boundary_outputs = set()
    ephemeral = set()

    for tid in produced_internally:
        consumed_by_internal_op = tid in consumed_internally

        if consumed_by_internal_op:
            # This tensor flows between ops inside the subgraph = ephemeral
            # It may also have external consumers, but those will be served
            # by recomputation or a different subgraph
            ephemeral.add(tid)
        else:
            # Not consumed internally at all = pure output of the subgraph
            boundary_outputs.add(tid)

    # Special case: tensors that are ephemeral but also graph outputs
    # or have external consumers still need to be trackable.
    # But for cost purposes, they are NOT evicted from this subgraph
    # unless they are in the boundary_outputs set.
    # If caller wants to retain an ephemeral tensor, it must appear 
    # in boundary_outputs. Check if any retained tensor is ephemeral
    # and promote it to boundary output (it needs to materialize).
    # Actually, retained tensors that are ephemeral don't make sense
    # in the current model - you can only retain outputs.

    return TensorClassification(
        boundary_inputs=non_resident_inputs,
        boundary_outputs=boundary_outputs,
        ephemeral=ephemeral,
        resident_inputs=resident_inputs,
    )


def compute_tile_geometry(
    graph: Graph,
    op_ids: list[int],
    config: Config,
) -> TileGeometry:
    """Compute the tiling structure for a subgraph."""
    # Determine output tensor dimensions
    # For a subgraph, all ops share the same spatial tiling
    # Use the output of the last op (or the subgraph's boundary output)
    output_w = 0
    output_h = 0
    reduction_dim = 0

    for oid in op_ids:
        op = graph.get_op(oid)
        for tid in op.output_tensor_ids:
            t = graph.get_tensor(tid)
            output_w = max(output_w, t.width)
            output_h = max(output_h, t.height)

        if op.op_type == OpType.MATMUL:
            # Reduction dim is the shared dimension between LHS cols and RHS rows
            # LHS: input[0], RHS: input[1]
            # For MatMul: LHS is (H x K), RHS is (K x W), output is (H x W)
            # The reduction dimension K = width of LHS = height of RHS
            lhs_tensor = graph.get_tensor(op.input_tensor_ids[0])
            reduction_dim = max(reduction_dim, lhs_tensor.width)

    num_tiles_w = math.ceil(output_w / config.w)
    num_tiles_h = math.ceil(output_h / config.h)
    total_spatial_tiles = num_tiles_w * num_tiles_h

    if reduction_dim > 0:
        num_k_steps = math.ceil(reduction_dim / config.k)
    else:
        num_k_steps = 1

    return TileGeometry(
        num_tiles_w=num_tiles_w,
        num_tiles_h=num_tiles_h,
        total_spatial_tiles=total_spatial_tiles,
        num_k_steps=num_k_steps,
        total_iterations=total_spatial_tiles * num_k_steps,
        output_w=output_w,
        output_h=output_h,
        reduction_dim=reduction_dim,
    )


def compute_working_set(
    graph: Graph,
    tensor_class: TensorClassification,
    config: Config,
    op_ids: list[int],
    state: ScheduleState,
) -> float:
    """Compute peak working set for memory validation."""
    ws = 0.0

    # Determine if we have split-K
    has_matmul = any(
        graph.get_op(oid).op_type == OpType.MATMUL for oid in op_ids
    )

    # Compute reduction dimension for k-step count
    reduction_dim = 0
    for oid in op_ids:
        op = graph.get_op(oid)
        if op.op_type == OpType.MATMUL:
            lhs = graph.get_tensor(op.input_tensor_ids[0])
            reduction_dim = max(reduction_dim, lhs.width)

    num_k_steps = math.ceil(reduction_dim / config.k) if reduction_dim > 0 else 1
    is_split_k = num_k_steps > 1

    # Input slices
    for tid, role in tensor_class.boundary_inputs.items():
        t = graph.get_tensor(tid)
        if role == "lhs":
            if is_split_k:
                # Split-K: LHS stored fully
                ws += t.size
            else:
                ws += config.k * config.h  # LHS slice
        elif role == "rhs":
            ws += config.w * config.k  # RHS slice (streamed)
        else:  # pointwise
            ws += config.w * config.h

    # Resident input slices (already in fast memory, still consume capacity)
    for tid in tensor_class.resident_inputs:
        # Resident tensors occupy their full size, not sliced
        t = graph.get_tensor(tid)
        ws += t.size

    # Output slices (accumulator for MatMul, output for Pointwise)
    # For the output, we hold one spatial tile at a time
    has_matmul = any(
        graph.get_op(oid).op_type == OpType.MATMUL for oid in op_ids
    )

    output_slice_size = config.w * config.h
    # Count unique boundary output tensors (usually 1)
    for tid in tensor_class.boundary_outputs:
        ws += output_slice_size

    # Retained tensors from previous steps that are still in fast memory
    # (but not already counted as resident inputs)
    for tid in state.tensors_in_fast_memory:
        if tid not in tensor_class.resident_inputs:
            # This tensor is just sitting in fast memory, occupying space
            t = graph.get_tensor(tid)
            ws += t.size

    return ws


def compute_subgraph_latency(
    graph: Graph,
    action: Action,
    state: ScheduleState,
) -> SubgraphLatencyResult:
    """
    Compute the total latency for executing a subgraph.
    This is the core physics engine of the environment.
    """
    config = action.config
    op_ids = action.operation_ids
    bw = graph.hardware.slow_memory_bandwidth
    native_w, native_h = graph.hardware.native_granularity

    # Step 1: Classify tensors
    tensor_class = classify_tensors(graph, op_ids, state)

    # Step 2: Compute tile geometry
    geom = compute_tile_geometry(graph, op_ids, config)

    # Step 3: Check working set
    ws = compute_working_set(graph, tensor_class, config, op_ids, state)
    if ws > graph.hardware.fast_memory_capacity:
        return SubgraphLatencyResult(
            total_latency=0.0,
            tile_latencies=[],
            working_set=ws,
            is_valid=False,
            error=f"OOM: working set {ws:.0f} > capacity {graph.hardware.fast_memory_capacity}",
        )

    # Step 4: Compute per-op costs
    pointwise_cost = sum(
        graph.get_op(oid).base_cost
        for oid in op_ids
        if graph.get_op(oid).op_type == OpType.POINTWISE
    )
    matmul_cost_per_k_step = sum(
        graph.get_op(oid).base_cost / geom.num_k_steps
        for oid in op_ids
        if graph.get_op(oid).op_type == OpType.MATMUL
    )

    # Step 5: Determine traversal order
    traversal = action.traversal_order
    if traversal is None:
        traversal = list(range(geom.total_spatial_tiles))

    # Step 6: Iterate tiles and compute latencies
    tile_latencies = []

    for tile_idx_in_order, tile_flat in enumerate(traversal):
        # Convert flat index to (tile_w_idx, tile_h_idx)
        tile_w_idx = tile_flat % geom.num_tiles_w
        tile_h_idx = tile_flat // geom.num_tiles_w

        # Previous tile for reuse detection
        if tile_idx_in_order > 0:
            prev_flat = traversal[tile_idx_in_order - 1]
            prev_tile_w = prev_flat % geom.num_tiles_w
            prev_tile_h = prev_flat // geom.num_tiles_w
        else:
            prev_tile_w = -1
            prev_tile_h = -1

        for k_step in range(geom.num_k_steps):
            # --- Memory In ---
            mem_in = 0.0
            is_first_k_step = (k_step == 0)
            is_first_tile = (tile_idx_in_order == 0)
            is_first_iteration = is_first_tile and is_first_k_step

            for tid, role in tensor_class.boundary_inputs.items():
                t = graph.get_tensor(tid)

                if role == "pointwise":
                    # Pointwise: load slice every new spatial tile, not on k-steps
                    if is_first_k_step:
                        slice_size = config.w * config.h
                        mem_in += slice_size / bw

                elif role == "lhs":
                    # LHS behavior depends on whether we have split-K
                    if geom.num_k_steps > 1:
                        # Split-K: load FULL tensor on first iteration, reuse across k-steps
                        if is_first_iteration:
                            mem_in += t.size / bw
                        # On subsequent k-steps AND tiles: already resident, reused
                    else:
                        # No split-K: load LHS strip per spatial tile
                        # LHS strip: [k x h], reused when tile_h is same
                        lhs_slice_size = config.k * config.h
                        if is_first_iteration:
                            mem_in += lhs_slice_size / bw
                        elif tile_h_idx == prev_tile_h:
                            pass  # Reused: same row
                        else:
                            mem_in += lhs_slice_size / bw

                elif role == "rhs":
                    # RHS is always streamed along k-dimension
                    rhs_slice_size = config.w * config.k
                    if geom.num_k_steps > 1:
                        # Split-K: load new RHS strip every k-step
                        if is_first_iteration:
                            mem_in += rhs_slice_size / bw
                        elif not is_first_k_step:
                            # New k-step: new RHS strip
                            mem_in += rhs_slice_size / bw
                        # New spatial tile, first k-step: reload RHS strip
                        elif is_first_k_step and not is_first_tile:
                            mem_in += rhs_slice_size / bw
                    else:
                        # No split-K: load RHS strip per spatial tile
                        # RHS strip: [w x k], reused when tile_w is same
                        if is_first_iteration:
                            mem_in += rhs_slice_size / bw
                        elif tile_w_idx == prev_tile_w:
                            pass  # Reused: same column
                        else:
                            mem_in += rhs_slice_size / bw

            # Resident inputs: zero load cost (already in fast memory)
            # But for MatMul, resident LHS/RHS may have different behavior
            # For now, resident = full tensor in fast memory, always available
            # (This handles the T0 reuse in Example 5 k-steps 2-4)

            # --- Memory Out ---
            mem_out = 0.0

            is_last_k_step = (k_step == geom.num_k_steps - 1)
            is_last_tile = (tile_idx_in_order == len(traversal) - 1)

            if is_last_k_step:
                # Output slice can be evicted (or retained)
                for tid in tensor_class.boundary_outputs:
                    if tid in action.tensors_to_retain:
                        # Retained: no eviction cost
                        # But only skip if this is NOT a graph output that must go to slow mem
                        # Actually, retain means keep in fast memory.
                        # Graph outputs must eventually go to slow memory,
                        # but that happens when the episode ends or a later subgraph evicts.
                        # For now, if retained, no eviction cost this step.
                        pass
                    else:
                        output_slice_size = config.w * config.h
                        mem_out += output_slice_size / bw

            # --- Compute ---
            compute = pointwise_cost + matmul_cost_per_k_step

            # --- Tile Latency (Roofline) ---
            tile_lat = max(compute, mem_in + mem_out)
            tile_latencies.append(tile_lat)

    total_latency = sum(tile_latencies)

    return SubgraphLatencyResult(
        total_latency=total_latency,
        tile_latencies=tile_latencies,
        working_set=ws,
        is_valid=True,
    )


def compute_naive_latency(graph: Graph) -> float:
    """
    Compute the total latency if every op is scheduled individually
    with native granularity, no fusion, no retention.
    This is the worst-case baseline for grading.
    """
    bw = graph.hardware.slow_memory_bandwidth
    native_w, native_h = graph.hardware.native_granularity
    total = 0.0

    for op in graph.operations:
        # Each op in its own subgraph, config = native granularity
        # All inputs loaded from slow memory, all outputs evicted

        # Determine output dimensions
        out_w = max(graph.get_tensor(tid).width for tid in op.output_tensor_ids)
        out_h = max(graph.get_tensor(tid).height for tid in op.output_tensor_ids)

        num_tiles_w = math.ceil(out_w / native_w)
        num_tiles_h = math.ceil(out_h / native_h)
        total_tiles = num_tiles_w * num_tiles_h

        if op.op_type == OpType.MATMUL:
            lhs = graph.get_tensor(op.input_tensor_ids[0])
            rhs = graph.get_tensor(op.input_tensor_ids[1])
            K = lhs.width
            # Use full K (native), so 1 k-step
            num_k = 1
            k_val = K

            for tile_i in range(total_tiles):
                tile_w_idx = tile_i % num_tiles_w
                tile_h_idx = tile_i // num_tiles_w

                # LHS slice: k x native_h
                lhs_size = K * native_h
                # RHS slice: native_w x k
                rhs_size = native_w * K
                # Output slice
                out_size = native_w * native_h

                # Reuse logic for naive (raster order)
                mem_in = 0.0
                if tile_i == 0:
                    mem_in = (lhs_size + rhs_size) / bw
                elif tile_w_idx == 0:
                    # New row: reload both
                    mem_in = (lhs_size + rhs_size) / bw
                else:
                    # Same row: LHS reused, reload RHS
                    mem_in = rhs_size / bw

                mem_out = out_size / bw
                compute = op.base_cost
                total += max(compute, mem_in + mem_out)
        else:
            # Pointwise
            for tile_i in range(total_tiles):
                mem_in = sum(
                    native_w * native_h / bw
                    for tid in op.input_tensor_ids
                )
                mem_out = sum(
                    native_w * native_h / bw
                    for tid in op.output_tensor_ids
                )
                compute = op.base_cost
                total += max(compute, mem_in + mem_out)

    return total
