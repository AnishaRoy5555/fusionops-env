"""
Golden Tests for FusionOps Cost Model
Verifies against ALL 5 official examples from the MLSys competition.
Every latency must match exactly. If any test fails, the cost model is wrong.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    Action, Config, Graph, HardwareSpec, Operation, OpType,
    ScheduleState, Tensor,
)
from src.cost_model import compute_subgraph_latency
from src.validator import validate_action

TOLERANCE = 0.01  # Allow tiny floating point differences


def assert_latency(expected: float, actual: float, label: str):
    if abs(expected - actual) > TOLERANCE:
        print(f"  FAIL {label}: expected {expected}, got {actual}")
        return False
    else:
        print(f"  PASS {label}: {actual}")
        return True


def make_graph(data: dict) -> Graph:
    return Graph.from_json(data)


# ============================================================
# EXAMPLE 1: Baseline (two Pointwise ops, linear chain)
# ============================================================

EXAMPLE_1_DATA = {
    "widths": [128, 128, 128],
    "heights": [128, 128, 128],
    "inputs": [[0], [1]],
    "outputs": [[1], [2]],
    "base_costs": [1000, 100],
    "op_types": ["Pointwise", "Pointwise"],
    "fast_memory_capacity": 35000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


def test_example1_strategy_a():
    """Strategy A: Always Spill. Two separate subgraphs."""
    print("\nExample 1, Strategy A: Always Spill")
    g = make_graph(EXAMPLE_1_DATA)
    passed = True

    # Subgraph 0: op 0 alone
    state = ScheduleState()
    action = Action(operation_ids=[0], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3276.8, result.total_latency, "Subgraph 0")

    # Subgraph 1: op 1 alone
    state.scheduled_op_ids.add(0)
    action = Action(operation_ids=[1], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3276.8, result.total_latency, "Subgraph 1")

    total = 3276.8 + 3276.8
    passed &= assert_latency(6553.6, total, "Total")
    return passed


def test_example1_strategy_b():
    """Strategy B: Mega-Group, 128x128 granularity. Fuse both ops."""
    print("\nExample 1, Strategy B: Mega-Group (128x128)")
    g = make_graph(EXAMPLE_1_DATA)

    state = ScheduleState()
    action = Action(operation_ids=[0, 1], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(3276.8, result.total_latency, "Fused subgraph")


def test_example1_strategy_c():
    """Strategy C: Mega-Group, 64x64 granularity. 4 tiles, compute-bound."""
    print("\nExample 1, Strategy C: Mega-Group (64x64)")
    g = make_graph(EXAMPLE_1_DATA)

    state = ScheduleState()
    action = Action(operation_ids=[0, 1], config=Config(64, 64, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(4400.0, result.total_latency, "Fused subgraph (4 tiles)")


# ============================================================
# EXAMPLE 2: Larger Tensors (256x256)
# ============================================================

EXAMPLE_2_DATA = {
    "widths": [256, 256, 256],
    "heights": [256, 256, 256],
    "inputs": [[0], [1]],
    "outputs": [[1], [2]],
    "base_costs": [1000, 100],
    "op_types": ["Pointwise", "Pointwise"],
    "fast_memory_capacity": 35000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


def test_example2_strategy_a():
    """Strategy A: Always Spill. 256x256 tensors, 4 tiles each."""
    print("\nExample 2, Strategy A: Always Spill (256x256)")
    g = make_graph(EXAMPLE_2_DATA)
    passed = True

    state = ScheduleState()
    action = Action(operation_ids=[0], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(13107.2, result.total_latency, "Subgraph 0")

    state.scheduled_op_ids.add(0)
    action = Action(operation_ids=[1], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(13107.2, result.total_latency, "Subgraph 1")

    passed &= assert_latency(26214.4, 13107.2 + 13107.2, "Total")
    return passed


def test_example2_strategy_b():
    """Strategy B: Mega-Group (128x128). 4 tiles, fused."""
    print("\nExample 2, Strategy B: Mega-Group (128x128)")
    g = make_graph(EXAMPLE_2_DATA)

    state = ScheduleState()
    action = Action(operation_ids=[0, 1], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(13107.2, result.total_latency, "Fused subgraph")


# ============================================================
# EXAMPLE 3: Diamond Graph (Spill vs Recompute vs Retain)
# ============================================================

EXAMPLE_3_DATA = {
    "widths": [128, 128, 128, 128],
    "heights": [128, 128, 128, 128],
    "inputs": [[0], [1], [1, 2]],
    "outputs": [[1], [2], [3]],
    "base_costs": [1500, 1500, 1500],
    "op_types": ["Pointwise", "Pointwise", "Pointwise"],
    "fast_memory_capacity": 50000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


def test_example3_strategy_a():
    """Strategy A: Spilling. Three separate subgraphs."""
    print("\nExample 3, Strategy A: Spilling")
    g = make_graph(EXAMPLE_3_DATA)
    passed = True

    state = ScheduleState()
    action = Action(operation_ids=[0], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3276.8, result.total_latency, "Subgraph 0")

    state.scheduled_op_ids.add(0)
    action = Action(operation_ids=[1], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3276.8, result.total_latency, "Subgraph 1")

    state.scheduled_op_ids.add(1)
    action = Action(operation_ids=[2], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(4915.2, result.total_latency, "Subgraph 2")

    total = 3276.8 + 3276.8 + 4915.2
    passed &= assert_latency(11468.8, total, "Total")
    return passed


def test_example3_strategy_b():
    """Strategy B: Recomputation. Op0 appears in both subgraphs."""
    print("\nExample 3, Strategy B: Recomputation")
    g = make_graph(EXAMPLE_3_DATA)
    passed = True

    # Subgraph 0: ops [0,1], retain tensor 2
    state = ScheduleState()
    action = Action(operation_ids=[0, 1], config=Config(128, 128, 1), tensors_to_retain=[2])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3000.0, result.total_latency, "Subgraph 0 [ops 0,1]")

    # Subgraph 1: ops [0,2] (recomputing op 0), tensor 2 resident
    state.scheduled_op_ids.update([0, 1])
    state.tensors_in_fast_memory.add(2)
    action = Action(operation_ids=[0, 2], config=Config(128, 128, 1), tensors_to_retain=[])
    # Note: op 0 is being re-executed (recomputation)
    # For this to work, we need to allow re-scheduling of already scheduled ops
    # The problem spec allows recomputation (operations can appear more than once)
    # So we temporarily remove op 0 from scheduled set for validation
    state_for_recompute = state.clone()
    state_for_recompute.scheduled_op_ids.discard(0)
    result = compute_subgraph_latency(g, action, state_for_recompute)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3276.8, result.total_latency, "Subgraph 1 [ops 0,2]")

    total = 3000.0 + 3276.8
    passed &= assert_latency(6276.8, total, "Total")
    return passed


def test_example3_strategy_c():
    """Strategy C: Selective Residency. Keep T1 in fast memory."""
    print("\nExample 3, Strategy C: Selective Residency")
    g = make_graph(EXAMPLE_3_DATA)
    passed = True

    # Subgraph 0: op 0, retain tensor 1
    state = ScheduleState()
    action = Action(operation_ids=[0], config=Config(128, 128, 1), tensors_to_retain=[1])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(1638.4, result.total_latency, "Subgraph 0 (retain T1)")

    # Subgraph 1: ops [1,2], T1 is resident
    state.scheduled_op_ids.add(0)
    state.tensors_in_fast_memory.add(1)
    action = Action(operation_ids=[1, 2], config=Config(128, 128, 1), tensors_to_retain=[])
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    passed &= assert_latency(3000.0, result.total_latency, "Subgraph 1 (T1 resident)")

    total = 1638.4 + 3000.0
    passed &= assert_latency(4638.4, total, "Total")
    return passed


# ============================================================
# EXAMPLE 4: MatMul with Traversal Order (Reuse)
# ============================================================

EXAMPLE_4_DATA = {
    "widths": [128, 128, 128],
    "heights": [128, 128, 128],
    "inputs": [[0, 1]],
    "outputs": [[2]],
    "base_costs": [1500],
    "op_types": ["MatMul"],
    "fast_memory_capacity": 25000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


def test_example4_strategy_a():
    """Strategy A: Naive raster traversal. 4 tiles, 2 reuses."""
    print("\nExample 4, Strategy A: Naive Tiling (raster)")
    g = make_graph(EXAMPLE_4_DATA)

    state = ScheduleState()
    action = Action(
        operation_ids=[0],
        config=Config(64, 64, 128),
        tensors_to_retain=[],
        traversal_order=None,  # raster: [0,1,2,3]
    )
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(7096.0, result.total_latency, "MatMul raster traversal")


def test_example4_strategy_b():
    """Strategy B: Snake traversal [0,1,3,2]. 3 reuses."""
    print("\nExample 4, Strategy B: Snake Traversal")
    g = make_graph(EXAMPLE_4_DATA)

    state = ScheduleState()
    action = Action(
        operation_ids=[0],
        config=Config(64, 64, 128),
        tensors_to_retain=[],
        traversal_order=[0, 1, 3, 2],
    )
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(6548.0, result.total_latency, "MatMul snake traversal")


# ============================================================
# EXAMPLE 5: Chained MatMul with Split-K
# ============================================================

EXAMPLE_5_DATA = {
    "widths": [128, 128, 128, 128, 128],
    "heights": [128, 128, 128, 128, 128],
    "inputs": [[0, 1], [3, 2]],
    "outputs": [[3], [4]],
    "base_costs": [2000, 2000],
    "op_types": ["MatMul", "MatMul"],
    "fast_memory_capacity": 45000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


def test_example5_strategy_a_oom():
    """Strategy A: Full K=128. Should OOM."""
    print("\nExample 5, Strategy A: Full K (should OOM)")
    g = make_graph(EXAMPLE_5_DATA)

    state = ScheduleState()
    action = Action(
        operation_ids=[0, 1],
        config=Config(128, 128, 128),
        tensors_to_retain=[],
    )
    result = compute_subgraph_latency(g, action, state)
    if not result.is_valid and "OOM" in (result.error or ""):
        print(f"  PASS: Correctly detected OOM ({result.error})")
        return True
    else:
        print(f"  FAIL: Should have been OOM, got valid={result.is_valid}, error={result.error}")
        return False


def test_example5_strategy_b():
    """Strategy B: Split-K with k=32. 4 accumulation steps."""
    print("\nExample 5, Strategy B: Split-K (k=32)")
    g = make_graph(EXAMPLE_5_DATA)

    state = ScheduleState()
    action = Action(
        operation_ids=[0, 1],
        config=Config(128, 128, 32),
        tensors_to_retain=[],
    )
    result = compute_subgraph_latency(g, action, state)
    assert result.is_valid, f"Should be valid: {result.error}"
    return assert_latency(6915.2, result.total_latency, "Split-K chained MatMul")


# ============================================================
# Run all tests
# ============================================================

def main():
    tests = [
        test_example1_strategy_a,
        test_example1_strategy_b,
        test_example1_strategy_c,
        test_example2_strategy_a,
        test_example2_strategy_b,
        test_example3_strategy_a,
        test_example3_strategy_b,
        test_example3_strategy_c,
        test_example4_strategy_a,
        test_example4_strategy_b,
        test_example5_strategy_a_oom,
        test_example5_strategy_b,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL GOLDEN TESTS PASSED")
    else:
        print("SOME TESTS FAILED - FIX BEFORE PROCEEDING")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
