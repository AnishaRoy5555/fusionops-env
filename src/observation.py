"""
FusionOps Observation Formatter
LLM-optimized observation with action hints, error feedback, and progress signal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .models import Graph, OpType, ScheduleState, Action, Config


def format_observation(
    graph: Graph,
    state: ScheduleState,
    last_action_result: str = "",
    last_action_error: Optional[dict] = None,
    max_steps: int = 20,
    naive_latency: float = 0.0,
) -> str:
    """
    Format the current environment state as LLM-friendly text.

    Args:
        graph: The computation graph.
        state: Current schedule state.
        last_action_result: Text description of last action result (for valid actions).
        last_action_error: Dict with 'type' and 'reason' if last action failed.
        max_steps: Maximum allowed steps.
        naive_latency: Naive baseline latency for progress estimation.
    """
    hw = graph.hardware
    lines = []

    # ============================================================
    # 1. LAST ACTION RESULT (only if there was an error - LLM attention front-loaded)
    # ============================================================
    if last_action_error:
        lines.append("=== LAST ACTION RESULT ===")
        lines.append("Status: INVALID")
        lines.append(f"Error Type: {last_action_error.get('type', 'Unknown Error')}")
        lines.append(f"Reason: {last_action_error.get('reason', 'No details')}")

        fix_hint = last_action_error.get('fix_hint')
        if fix_hint:
            lines.append("")
            lines.append("Fix Hint:")
            lines.append(f"  {fix_hint}")
        lines.append("")
    elif last_action_result:
        lines.append("=== LAST ACTION RESULT ===")
        lines.append(f"Status: {last_action_result}")
        lines.append("")

    # ============================================================
    # 2. CURRENT STATE
    # ============================================================
    lines.append("=== CURRENT STATE ===")
    completed_ops = sorted(state.scheduled_op_ids)
    ready_ops = _find_ready_ops(graph, state)
    lines.append(f"Completed Ops: {completed_ops}")
    lines.append(f"Ready Ops: {ready_ops}")
    lines.append(f"Step: {state.step_count}/{max_steps}")
    lines.append("")

    # ============================================================
    # 3. MEMORY
    # ============================================================
    lines.append("=== MEMORY ===")
    if state.tensors_in_fast_memory:
        used = sum(graph.get_tensor(tid).size for tid in state.tensors_in_fast_memory)
        tensor_list = sorted(state.tensors_in_fast_memory)
        lines.append(f"Fast Memory Tensors: {tensor_list}")
        lines.append(f"Capacity Used: {used} / {hw.fast_memory_capacity}")
    else:
        lines.append("Fast Memory: empty")
        lines.append(f"Capacity: 0 / {hw.fast_memory_capacity}")
    lines.append(f"Slow Memory Bandwidth: {hw.slow_memory_bandwidth}")
    lines.append(f"Native Granularity: {hw.native_granularity[0]}x{hw.native_granularity[1]}")
    lines.append("")

    # ============================================================
    # 4. GRAPH SUMMARY
    # ============================================================
    lines.append("=== GRAPH SUMMARY ===")
    for op in graph.operations:
        status = "DONE" if op.id in state.scheduled_op_ids else "TODO"
        in_str = ",".join(f"T{tid}" for tid in op.input_tensor_ids)
        out_str = ",".join(f"T{tid}" for tid in op.output_tensor_ids)
        lines.append(f"  Op{op.id} [{status}] {op.op_type.value}: [{in_str}]->[{out_str}] cost={op.base_cost}")
    lines.append("")

    # Tensor info (compact)
    lines.append("Tensors:")
    for t in graph.tensors:
        loc = ""
        if t.id in graph.graph_input_tensor_ids:
            loc = "slow_mem (graph input)"
        elif t.id in state.tensors_in_fast_memory:
            loc = "fast_mem"
        elif t.id in graph.tensor_producer:
            prod = graph.tensor_producer[t.id]
            if prod in state.scheduled_op_ids:
                loc = f"slow_mem (from Op{prod})"
            else:
                loc = f"not_yet_computed (Op{prod})"
        lines.append(f"  T{t.id}: {t.width}x{t.height} ({loc})")
    lines.append("")

    # ============================================================
    # 5. VALID ACTION EXAMPLES
    # ============================================================
    hints = _generate_action_hints(graph, state)
    lines.append("=== VALID ACTION EXAMPLES (use as templates) ===")
    for i, hint in enumerate(hints, 1):
        lines.append(f"{i}. {hint}")
    lines.append("")

    # ============================================================
    # 6. CONSTRAINTS
    # ============================================================
    lines.append("=== CONSTRAINTS ===")
    lines.append("- ops MUST be from READY OPS list above")
    lines.append("- retain MUST only contain output tensors of the chosen ops")
    lines.append("- For Pointwise: use config=[128,128,1]")
    lines.append("- For MatMul: use config=[128,128,K] where K is the reduction dim")
    lines.append("- Working set must fit in fast memory or you get OOM")
    lines.append("")

    # ============================================================
    # 7. BEST PRACTICES (general guidance, not task-specific)
    # ============================================================
    lines.append("=== BEST PRACTICES ===")
    lines.append("- Prefer fusing connected ops to make intermediate tensors ephemeral")
    lines.append("- Fusion can often be extended beyond 2-3 ops if memory allows")
    lines.append("- Retain tensors that will be used in the very next step")
    lines.append("- Use native granularity unless memory forces smaller tiles")
    lines.append("- For MatMul fused with another op, consider split-K (smaller k) to fit memory")
    lines.append("")

    # ============================================================
    # 8. PROGRESS
    # ============================================================
    lines.append("=== PROGRESS ===")
    total_ops = len(graph.operations)
    completed = len(state.scheduled_op_ids)
    lines.append(f"Completed: {completed}/{total_ops} ops")
    lines.append(f"Current latency: {state.total_latency:.1f}")
    if naive_latency > 0:
        if state.total_latency > 0:
            efficiency = (naive_latency - state.total_latency) / naive_latency * 100
            lines.append(f"Naive baseline: {naive_latency:.1f}")
            lines.append(f"Improvement vs naive: {efficiency:+.1f}%")
        else:
            lines.append(f"Naive baseline: {naive_latency:.1f}")
    lines.append("")

    # ============================================================
    # 9. ACTION FORMAT REMINDER
    # ============================================================
    lines.append("=== ACTION FORMAT ===")
    lines.append("SCHEDULE ops=[op_ids] config=[w,h,k] retain=[tensor_ids]")

    return "\n".join(lines)


def _find_ready_ops(graph: Graph, state: ScheduleState) -> List[int]:
    """Find ops whose predecessors are all scheduled."""
    ready = []
    for op in graph.operations:
        if op.id in state.scheduled_op_ids:
            continue
        preds = graph.op_predecessors.get(op.id, set())
        if preds.issubset(state.scheduled_op_ids):
            ready.append(op.id)
    return ready


def _validate_hint(graph: Graph, state: ScheduleState, action_str: str) -> bool:
    """
    Test if an action string is actually valid given current state.
    Runs it through the parser, validator, and cost model.
    Returns True only if it would not produce any error.
    """
    try:
        # Parse the action
        import re as _re
        ops_match = _re.search(r'ops\s*=\s*\[([^\]]*)\]', action_str)
        config_match = _re.search(r'config\s*=\s*\[(\d+),(\d+),(\d+)\]', action_str)
        retain_match = _re.search(r'retain\s*=\s*\[([^\]]*)\]', action_str)
        if not ops_match or not config_match:
            return False

        ops_str = ops_match.group(1).strip()
        if not ops_str:
            return False
        op_ids = [int(x.strip()) for x in ops_str.split(",") if x.strip()]
        config = Config(
            int(config_match.group(1)),
            int(config_match.group(2)),
            int(config_match.group(3)),
        )
        retain_str = retain_match.group(1).strip() if retain_match else ""
        retain = [int(x.strip()) for x in retain_str.split(",") if x.strip()] if retain_str else []

        action = Action(
            operation_ids=op_ids,
            config=config,
            tensors_to_retain=retain,
        )

        # Run validator
        from .validator import validate_action
        from .cost_model import compute_subgraph_latency

        # Allow recomputation (clone state and remove ops being scheduled)
        test_state = state.clone()
        for oid in op_ids:
            test_state.scheduled_op_ids.discard(oid)

        validation = validate_action(graph, action, test_state)
        if not validation.is_valid:
            return False

        # Run cost model to check OOM
        result = compute_subgraph_latency(graph, action, test_state)
        if not result.is_valid:
            return False

        return True
    except Exception:
        return False


def _generate_action_hints(graph: Graph, state: ScheduleState) -> List[str]:
    """
    Generate 2-4 syntactically valid action examples.
    Each hint is validated against the actual cost model and validator.
    Only hints that would actually succeed are returned.
    """
    ready_ops = _find_ready_ops(graph, state)
    candidate_hints = []

    if not ready_ops:
        return ["SCHEDULE ops=[0] config=[128,128,1] retain=[]"]

    first_op = graph.get_op(ready_ops[0])

    # PRIORITY 1: Pair fusion (smallest fusion example)
    fusion_pair = _find_fusion_pair(graph, ready_ops, state)
    if fusion_pair:
        a, b = fusion_pair
        op_a = graph.get_op(a)
        op_b = graph.get_op(b)
        if op_a.op_type == OpType.POINTWISE and op_b.op_type == OpType.POINTWISE:
            candidate_hints.append(f"SCHEDULE ops=[{a},{b}] config=[128,128,1] retain=[]")
        else:
            candidate_hints.append(f"SCHEDULE ops=[{a},{b}] config=[128,128,32] retain=[]")

    # PRIORITY 2: 3-op chain (shows fusion can extend, capped to avoid leaking optimal)
    # Pattern: 2-op then 3-op suggests "this can extend further"
    fusion_chain = _find_fusion_chain(graph, ready_ops, state)
    if len(fusion_chain) >= 3:
        chain_to_show = fusion_chain[:3]
        has_matmul = any(graph.get_op(oid).op_type == OpType.MATMUL for oid in chain_to_show)
        ops_str = ",".join(str(oid) for oid in chain_to_show)
        if has_matmul:
            candidate_hints.append(f"SCHEDULE ops=[{ops_str}] config=[128,128,32] retain=[]")
        else:
            candidate_hints.append(f"SCHEDULE ops=[{ops_str}] config=[128,128,1] retain=[]")

    # PRIORITY 3: Single op (always valid baseline)
    if first_op.op_type == OpType.MATMUL:
        lhs = graph.get_tensor(first_op.input_tensor_ids[0])
        K = lhs.width
        candidate_hints.append(f"SCHEDULE ops=[{first_op.id}] config=[128,128,{K}] retain=[]")
    else:
        candidate_hints.append(f"SCHEDULE ops=[{first_op.id}] config=[128,128,1] retain=[]")

    # PRIORITY 4: Single op with retention (for cases where downstream needs it)
    if len(first_op.output_tensor_ids) > 0:
        retain_tid = first_op.output_tensor_ids[0]
        if retain_tid in graph.tensor_consumers and graph.tensor_consumers[retain_tid]:
            if first_op.op_type == OpType.MATMUL:
                lhs = graph.get_tensor(first_op.input_tensor_ids[0])
                K = lhs.width
                candidate_hints.append(f"SCHEDULE ops=[{first_op.id}] config=[128,128,{K}] retain=[{retain_tid}]")
            else:
                candidate_hints.append(f"SCHEDULE ops=[{first_op.id}] config=[128,128,1] retain=[{retain_tid}]")

    # PRIORITY 5: Smaller tile for memory-tight cases
    if first_op.op_type == OpType.MATMUL:
        lhs = graph.get_tensor(first_op.input_tensor_ids[0])
        K = lhs.width
        candidate_hints.append(f"SCHEDULE ops=[{first_op.id}] config=[64,64,{K}] retain=[]")

    # VALIDATE every candidate against actual cost model
    valid_hints = []
    for h in candidate_hints:
        if _validate_hint(graph, state, h):
            valid_hints.append(h)

    # Always guarantee at least one hint
    if not valid_hints:
        # Fallback: try reduced sizes for the first op
        if first_op.op_type == OpType.MATMUL:
            lhs = graph.get_tensor(first_op.input_tensor_ids[0])
            for k in [32, 16, 8]:
                fallback = f"SCHEDULE ops=[{first_op.id}] config=[64,64,{k}] retain=[]"
                if _validate_hint(graph, state, fallback):
                    valid_hints.append(fallback)
                    break
        if not valid_hints:
            # Last resort: just show something
            valid_hints.append(candidate_hints[0] if candidate_hints else
                f"SCHEDULE ops=[{ready_ops[0]}] config=[128,128,1] retain=[]")

    return valid_hints[:4]


def _find_fusion_pair(graph: Graph, ready_ops: List[int], state: ScheduleState = None) -> Optional[Tuple[int, int]]:
    """
    Find a producer-consumer pair where:
    - producer is in ready_ops (or already scheduled and being recomputed)
    - consumer's other dependencies are satisfied or are graph inputs
    
    The consumer doesn't need to be in ready_ops because fusing means scheduling 
    them together in the same step.
    """
    ready_set = set(ready_ops)
    scheduled = state.scheduled_op_ids if state else set()
    
    for op_id in ready_ops:
        op = graph.get_op(op_id)
        for tid in op.output_tensor_ids:
            if tid in graph.tensor_consumers:
                for consumer_id in graph.tensor_consumers[tid]:
                    if consumer_id == op_id:
                        continue
                    if consumer_id in scheduled:
                        continue
                    # Check consumer's OTHER inputs (besides tid which we're producing)
                    consumer = graph.get_op(consumer_id)
                    other_deps_ok = True
                    for dep_tid in consumer.input_tensor_ids:
                        if dep_tid == tid:
                            continue
                        # Other dep must be either: already scheduled, or a graph input
                        if dep_tid in graph.graph_input_tensor_ids:
                            continue
                        if dep_tid in graph.tensor_producer:
                            producer = graph.tensor_producer[dep_tid]
                            if producer not in scheduled:
                                other_deps_ok = False
                                break
                    if other_deps_ok:
                        return (op_id, consumer_id)
    return None


def _find_fusion_chain(graph: Graph, ready_ops: List[int], state: ScheduleState = None) -> List[int]:
    """
    Find the longest connected chain of ops starting from a ready op.
    Returns list of op IDs in execution order. Length 1 if no chain found.
    
    A chain is: op_a produces a tensor consumed by op_b, op_b produces a tensor 
    consumed by op_c, etc. Each op's other dependencies must be satisfied.
    """
    if not ready_ops:
        return []

    scheduled = set(state.scheduled_op_ids) if state else set()

    # BFS to find the longest chain starting from each ready op
    best_chain = []

    for start_op in ready_ops:
        chain = [start_op]
        chain_set = {start_op}

        # Greedy extension: at each step, try to add an op whose only unsatisfied 
        # dependency is the previous chain output
        while True:
            extended = False
            last_op = graph.get_op(chain[-1])

            # Find ops that consume one of last_op's outputs
            for tid in last_op.output_tensor_ids:
                if tid not in graph.tensor_consumers:
                    continue
                for next_id in graph.tensor_consumers[tid]:
                    if next_id in chain_set:
                        continue
                    if next_id in scheduled:
                        continue
                    next_op = graph.get_op(next_id)

                    # Check if all of next_op's deps are: in chain, scheduled, or graph input
                    all_deps_ok = True
                    for dep_tid in next_op.input_tensor_ids:
                        if dep_tid in graph.graph_input_tensor_ids:
                            continue
                        if dep_tid in graph.tensor_producer:
                            producer = graph.tensor_producer[dep_tid]
                            if producer in chain_set:
                                continue
                            if producer in scheduled:
                                continue
                            all_deps_ok = False
                            break

                    if all_deps_ok:
                        chain.append(next_id)
                        chain_set.add(next_id)
                        extended = True
                        break
                if extended:
                    break

            if not extended:
                break

        if len(chain) > len(best_chain):
            best_chain = chain

    return best_chain
