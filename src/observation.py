"""
FusionOps Observation Formatter
Converts environment state into human/LLM-readable text.
"""

from __future__ import annotations

from .models import Graph, OpType, ScheduleState


def format_observation(
    graph: Graph,
    state: ScheduleState,
    last_action_result: str = "",
    max_steps: int = 20,
) -> str:
    """Format the current environment state as readable text for an LLM agent."""
    hw = graph.hardware
    lines = []

    # Header
    lines.append("=== FUSIONOPS SCHEDULING STATE ===")
    lines.append("")

    # Hardware
    lines.append("Hardware:")
    lines.append(f"  Fast Memory Capacity: {hw.fast_memory_capacity}")
    lines.append(f"  Slow Memory Bandwidth: {hw.slow_memory_bandwidth}")
    lines.append(f"  Native Granularity: {hw.native_granularity[0]}x{hw.native_granularity[1]}")
    lines.append("")

    # Last action result
    if last_action_result:
        lines.append(f"Last Action Result: {last_action_result}")
        lines.append("")

    # Progress
    total_ops = len(graph.operations)
    scheduled = len(state.scheduled_op_ids)
    remaining = total_ops - scheduled
    lines.append(f"Progress: {scheduled}/{total_ops} ops scheduled, Step {state.step_count}/{max_steps}")
    lines.append(f"Total Latency So Far: {state.total_latency:.1f}")
    lines.append("")

    # Fast memory contents
    if state.tensors_in_fast_memory:
        lines.append("Tensors in Fast Memory:")
        total_mem = 0
        for tid in sorted(state.tensors_in_fast_memory):
            t = graph.get_tensor(tid)
            lines.append(f"  T{tid} ({t.width}x{t.height}, size={t.size})")
            total_mem += t.size
        lines.append(f"  Total used: {total_mem} / {hw.fast_memory_capacity}")
    else:
        lines.append("Fast Memory: empty")
    lines.append("")

    # Graph structure (show once, condensed)
    lines.append("Computation Graph:")
    for op in graph.operations:
        status = "DONE" if op.id in state.scheduled_op_ids else "TODO"
        in_str = ", ".join(f"T{tid}" for tid in op.input_tensor_ids)
        out_str = ", ".join(f"T{tid}" for tid in op.output_tensor_ids)
        lines.append(f"  Op{op.id} [{status}] {op.op_type.value}: [{in_str}] -> [{out_str}] cost={op.base_cost}")
    lines.append("")

    # Tensor info
    lines.append("Tensors:")
    for t in graph.tensors:
        origin = ""
        if t.id in graph.graph_input_tensor_ids:
            origin = " (graph input, in slow memory)"
        elif t.id in graph.tensor_producer:
            prod_op = graph.tensor_producer[t.id]
            if prod_op in state.scheduled_op_ids:
                if t.id in state.tensors_in_fast_memory:
                    origin = f" (produced by Op{prod_op}, in fast memory)"
                else:
                    origin = f" (produced by Op{prod_op}, in slow memory)"
            else:
                origin = f" (produced by Op{prod_op}, not yet computed)"
        lines.append(f"  T{t.id}: {t.width}x{t.height}{origin}")
    lines.append("")

    # Ready operations (all predecessors scheduled)
    ready_ops = []
    for op in graph.operations:
        if op.id in state.scheduled_op_ids:
            continue
        preds = graph.op_predecessors.get(op.id, set())
        if preds.issubset(state.scheduled_op_ids):
            ready_ops.append(op.id)

    if ready_ops:
        lines.append(f"Ready to Schedule: {', '.join(f'Op{oid}' for oid in ready_ops)}")
    else:
        lines.append("Ready to Schedule: none (all ops scheduled or blocked)")
    lines.append("")

    # Possible fusion groups (hint: adjacent ready ops that share tensors)
    if len(ready_ops) > 1:
        fusion_hints = _find_fusion_candidates(graph, ready_ops, state)
        if fusion_hints:
            lines.append("Possible Fusion Groups:")
            for group in fusion_hints[:5]:  # limit to 5
                ops_str = ", ".join(f"Op{oid}" for oid in group)
                lines.append(f"  [{ops_str}]")
            lines.append("")

    # Schedule history
    if state.schedule_history:
        lines.append("Schedule History:")
        for i, entry in enumerate(state.schedule_history):
            ops_str = ", ".join(f"Op{oid}" for oid in entry.operation_ids)
            retain_str = ", ".join(f"T{tid}" for tid in entry.tensors_to_retain) if entry.tensors_to_retain else "none"
            lines.append(
                f"  Step {i+1}: ops=[{ops_str}] config=[{entry.config.w},{entry.config.h},{entry.config.k}] "
                f"retain=[{retain_str}] latency={entry.latency:.1f}"
            )
        lines.append("")

    return "\n".join(lines)


def _find_fusion_candidates(
    graph: Graph,
    ready_ops: list[int],
    state: ScheduleState,
) -> list[list[int]]:
    """Find pairs/triples of ready ops that could be fused (share tensors)."""
    ready_set = set(ready_ops)
    candidates = []

    # Check pairs: op A produces a tensor consumed by op B
    for oid_a in ready_ops:
        op_a = graph.get_op(oid_a)
        for tid in op_a.output_tensor_ids:
            if tid in graph.tensor_consumers:
                for oid_b in graph.tensor_consumers[tid]:
                    if oid_b in ready_set and oid_b != oid_a:
                        pair = sorted([oid_a, oid_b])
                        if pair not in candidates:
                            candidates.append(pair)

    # Also check: two ready ops that share the same input tensor
    # (can potentially be fused if one feeds the other through a chain)

    # Check triples: extend pairs
    extended = []
    for pair in candidates:
        for oid in ready_ops:
            if oid not in pair:
                # Check if oid connects to the pair
                op = graph.get_op(oid)
                pair_tensors = set()
                for pid in pair:
                    p_op = graph.get_op(pid)
                    pair_tensors.update(p_op.output_tensor_ids)
                    pair_tensors.update(p_op.input_tensor_ids)

                connected = any(
                    tid in pair_tensors
                    for tid in op.input_tensor_ids + op.output_tensor_ids
                )
                if connected:
                    triple = sorted(pair + [oid])
                    if triple not in extended and triple not in candidates:
                        extended.append(triple)

    return candidates + extended[:3]
