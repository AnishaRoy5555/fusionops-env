"""
FusionOps Task Definitions
Three fixed tasks with increasing difficulty.
Each returns a Graph ready for the environment.
"""

from __future__ import annotations

from .models import Graph


# ============================================================
# TASK 1: Linear Fusion Chain (Easy)
# ============================================================
# 6 Pointwise ops in a linear chain: T0 -> Op0 -> T1 -> Op1 -> T2 -> ... -> T6
# Optimal: fuse all into one mega-subgraph.
# Naive: 6 separate subgraphs, each loading and evicting.
# Tests: basic fusion understanding.
# Expected baseline score: 0.6-0.8

TASK_1_DATA = {
    "widths":  [128, 128, 128, 128, 128, 128, 128],
    "heights": [128, 128, 128, 128, 128, 128, 128],
    "inputs": [[0], [1], [2], [3], [4], [5]],
    "outputs": [[1], [2], [3], [4], [5], [6]],
    "base_costs": [800, 600, 900, 500, 700, 400],
    "op_types": ["Pointwise", "Pointwise", "Pointwise",
                 "Pointwise", "Pointwise", "Pointwise"],
    "fast_memory_capacity": 40000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


# ============================================================
# TASK 2: Diamond with Branching (Medium)
# ============================================================
# Graph structure:
#   T0 -> Op0 -> T1
#   T1 -> Op1 -> T2
#   T1 -> Op2 -> T3
#   T2, T3 -> Op3 -> T4
#   T4 -> Op4 -> T5
#   T1, T5 -> Op5 -> T6
#
# Key challenges:
# - T1 has 3 consumers (Op1, Op2, Op5) = high fan-out, retain decision critical
# - Diamond: Op1 and Op2 branch from T1, merge at Op3
# - Skip connection: T1 feeds Op5 directly (late consumer)
# - Agent must decide: retain T1 throughout? Or recompute Op0?
# Expected baseline score: 0.3-0.5

TASK_2_DATA = {
    "widths":  [128, 128, 128, 128, 128, 128, 128],
    "heights": [128, 128, 128, 128, 128, 128, 128],
    "inputs": [
        [0],        # Op0: T0 -> T1
        [1],        # Op1: T1 -> T2
        [1],        # Op2: T1 -> T3
        [2, 3],     # Op3: T2, T3 -> T4
        [4],        # Op4: T4 -> T5
        [1, 5],     # Op5: T1, T5 -> T6 (skip connection)
    ],
    "outputs": [[1], [2], [3], [4], [5], [6]],
    "base_costs": [1200, 1000, 1000, 1500, 800, 1100],
    "op_types": ["Pointwise", "Pointwise", "Pointwise",
                 "Pointwise", "Pointwise", "Pointwise"],
    "fast_memory_capacity": 50000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


# ============================================================
# TASK 3: Chained MatMul with Memory Pressure (Hard)
# ============================================================
# Graph structure:
#   T0 (128x128), T1 (128x128) -> Op0 (MatMul) -> T4 (128x128)
#   T4 (128x128), T2 (128x128) -> Op1 (MatMul) -> T5 (128x128)
#   T5 (128x128) -> Op2 (Pointwise) -> T6 (128x128)
#   T6 (128x128), T3 (128x128) -> Op3 (MatMul) -> T7 (128x128)
#
# Key challenges:
# - All tensors 128x128, fast memory = 50000
# - Single MatMul at native: WS = 3 * 16384 = 49152 < 50000 = OK (barely)
# - But fusing 2 MatMuls at native K: WS >> 50000 = OOM, needs split-K
# - Op2 (Pointwise) breaks the MatMul chain: fusion boundary decision
# - Mixed op types
# Expected baseline score: 0.1-0.3

TASK_3_DATA = {
    "widths":  [128, 128, 128, 128, 128, 128, 128, 128],
    "heights": [128, 128, 128, 128, 128, 128, 128, 128],
    "inputs": [
        [0, 1],     # Op0 (MatMul): T0 @ T1 -> T4
        [4, 2],     # Op1 (MatMul): T4 @ T2 -> T5
        [5],        # Op2 (Pointwise): T5 -> T6
        [6, 3],     # Op3 (MatMul): T6 @ T3 -> T7
    ],
    "outputs": [[4], [5], [6], [7]],
    "base_costs": [2000, 2000, 500, 2000],
    "op_types": ["MatMul", "MatMul", "Pointwise", "MatMul"],
    "fast_memory_capacity": 50000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


# ============================================================
# TASK 4: Multi-Stage MatMul with Skip Connection (Hardest)
# ============================================================
# Graph structure (8 ops, 16 tensors):
#   T0(128x128), T1(128x128) -> Op0 (MatMul) -> T8
#   T8 -> Op1 (Pointwise) -> T9
#   T9, T2(128x128) -> Op2 (MatMul) -> T10
#   T10 -> Op3 (Pointwise) -> T11
#   T11, T3(128x128) -> Op4 (MatMul) -> T12
#   T12 -> Op5 (Pointwise) -> T13
#   T13, T8 -> Op6 (Pointwise) -> T14   [skip connection from T8]
#   T14 -> Op7 (Pointwise) -> T15
#
# Key challenges:
# - 3 MatMul ops, each needs split-K when fused with subsequent Pointwise
# - T8 has 2 consumers far apart (Op1 and Op6) - tests retention reasoning
# - 8-op chain requires planning horizon
# - Naive scoring ~0.0, basic fusion ~0.12, smart fusion ~0.25, optimal ~0.30
# Expected baseline score: 0.05-0.20

TASK_4_DATA = {
    "widths":  [128] * 16,
    "heights": [128] * 16,
    "inputs": [
        [0, 1],     # Op0 (MatMul): T0 @ T1 -> T8
        [8],        # Op1 (Pointwise): T8 -> T9
        [9, 2],     # Op2 (MatMul): T9 @ T2 -> T10
        [10],       # Op3 (Pointwise): T10 -> T11
        [11, 3],    # Op4 (MatMul): T11 @ T3 -> T12
        [12],       # Op5 (Pointwise): T12 -> T13
        [13, 8],    # Op6 (Pointwise): T13 + T8 -> T14 [skip connection]
        [14],       # Op7 (Pointwise): T14 -> T15
    ],
    "outputs": [[8], [9], [10], [11], [12], [13], [14], [15]],
    "base_costs": [2000, 400, 2000, 400, 2000, 400, 600, 400],
    "op_types": ["MatMul", "Pointwise", "MatMul", "Pointwise",
                 "MatMul", "Pointwise", "Pointwise", "Pointwise"],
    "fast_memory_capacity": 60000,
    "slow_memory_bandwidth": 10,
    "native_granularity": [128, 128],
}


# ============================================================
# Task registry
# ============================================================

TASKS = {
    "task1_linear": TASK_1_DATA,
    "task2_diamond": TASK_2_DATA,
    "task3_matmul": TASK_3_DATA,
    "task4_multistage": TASK_4_DATA,
}


def load_task(task_name: str) -> Graph:
    """Load a task by name. Returns a Graph."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    return Graph.from_json(TASKS[task_name])


def get_task_config(task_name: str) -> dict:
    """Get task metadata for environment configuration."""
    configs = {
        "task1_linear": {"max_steps": 10, "description": "Linear chain of 6 Pointwise ops. Test basic fusion."},
        "task2_diamond": {"max_steps": 12, "description": "Diamond graph with skip connections. Test retention decisions."},
        "task3_matmul": {"max_steps": 15, "description": "Chained MatMuls with tight memory. Test split-K and memory management."},
        "task4_multistage": {"max_steps": 20, "description": "Multi-stage MatMul with skip connection. Test long-horizon planning, split-K, and selective retention."},
    }
    return configs[task_name]


def list_tasks() -> list[str]:
    return list(TASKS.keys())
