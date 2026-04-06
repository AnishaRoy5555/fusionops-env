"""
FusionOps Data Models
Core data structures for the ML graph scheduling environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OpType(str, Enum):
    MATMUL = "MatMul"
    POINTWISE = "Pointwise"


@dataclass
class Tensor:
    id: int
    width: int
    height: int

    @property
    def size(self) -> int:
        return self.width * self.height


@dataclass
class Operation:
    id: int
    op_type: OpType
    input_tensor_ids: list[int]  # For MatMul: [LHS, RHS]
    output_tensor_ids: list[int]
    base_cost: float


@dataclass
class HardwareSpec:
    fast_memory_capacity: int
    slow_memory_bandwidth: float
    native_granularity: tuple[int, int]  # (native_w, native_h)


@dataclass
class Graph:
    tensors: list[Tensor]
    operations: list[Operation]
    hardware: HardwareSpec

    # Derived (computed on load)
    graph_input_tensor_ids: set[int] = field(default_factory=set)
    graph_output_tensor_ids: set[int] = field(default_factory=set)
    tensor_producer: dict[int, int] = field(default_factory=dict)  # tensor_id -> op_id
    tensor_consumers: dict[int, list[int]] = field(default_factory=dict)  # tensor_id -> [op_ids]
    op_predecessors: dict[int, set[int]] = field(default_factory=dict)  # op_id -> {predecessor op_ids}
    op_successors: dict[int, set[int]] = field(default_factory=dict)  # op_id -> {successor op_ids}

    def __post_init__(self):
        self._derive_graph_structure()

    def _derive_graph_structure(self):
        produced_tensors = set()
        consumed_tensors = set()

        # Build producer/consumer maps
        for op in self.operations:
            self.op_predecessors[op.id] = set()
            self.op_successors[op.id] = set()

            for tid in op.output_tensor_ids:
                self.tensor_producer[tid] = op.id
                produced_tensors.add(tid)

            for tid in op.input_tensor_ids:
                if tid not in self.tensor_consumers:
                    self.tensor_consumers[tid] = []
                self.tensor_consumers[tid].append(op.id)
                consumed_tensors.add(tid)

        # Graph inputs: consumed but not produced
        self.graph_input_tensor_ids = consumed_tensors - produced_tensors

        # Graph outputs: produced but not consumed
        self.graph_output_tensor_ids = produced_tensors - consumed_tensors

        # Build op dependency edges
        for op in self.operations:
            for tid in op.input_tensor_ids:
                if tid in self.tensor_producer:
                    pred_op_id = self.tensor_producer[tid]
                    self.op_predecessors[op.id].add(pred_op_id)
                    self.op_successors[pred_op_id].add(op.id)

    def get_tensor(self, tid: int) -> Tensor:
        return self.tensors[tid]

    def get_op(self, oid: int) -> Operation:
        return self.operations[oid]

    @staticmethod
    def from_json(data: dict) -> Graph:
        """Load graph from Google's JSON format."""
        widths = data["widths"]
        heights = data["heights"]
        inputs = data["inputs"]
        outputs = data["outputs"]
        base_costs = data["base_costs"]
        op_types = data["op_types"]

        tensors = [
            Tensor(id=i, width=w, height=h)
            for i, (w, h) in enumerate(zip(widths, heights))
        ]

        operations = [
            Operation(
                id=i,
                op_type=OpType(op_types[i]),
                input_tensor_ids=inputs[i],
                output_tensor_ids=outputs[i],
                base_cost=base_costs[i],
            )
            for i in range(len(base_costs))
        ]

        ng = data["native_granularity"]
        hardware = HardwareSpec(
            fast_memory_capacity=data["fast_memory_capacity"],
            slow_memory_bandwidth=data["slow_memory_bandwidth"],
            native_granularity=(ng[0], ng[1]),
        )

        return Graph(tensors=tensors, operations=operations, hardware=hardware)

    @staticmethod
    def from_json_file(path: str) -> Graph:
        with open(path) as f:
            return Graph.from_json(json.load(f))


@dataclass
class Config:
    """Execution granularity [w, h, k]."""
    w: int
    h: int
    k: int


@dataclass
class Action:
    """Agent's action: schedule a subgraph with a config."""
    operation_ids: list[int]
    config: Config
    tensors_to_retain: list[int] = field(default_factory=list)
    traversal_order: Optional[list[int]] = None


@dataclass
class SubgraphEntry:
    """A completed schedule step."""
    operation_ids: list[int]
    config: Config
    tensors_to_retain: list[int]
    traversal_order: Optional[list[int]]
    latency: float


@dataclass
class ScheduleState:
    """Mutable state tracking the schedule being built."""
    scheduled_op_ids: set[int] = field(default_factory=set)
    tensors_in_fast_memory: set[int] = field(default_factory=set)
    schedule_history: list[SubgraphEntry] = field(default_factory=list)
    total_latency: float = 0.0
    step_count: int = 0

    def clone(self) -> ScheduleState:
        return ScheduleState(
            scheduled_op_ids=set(self.scheduled_op_ids),
            tensors_in_fast_memory=set(self.tensors_in_fast_memory),
            schedule_history=list(self.schedule_history),
            total_latency=self.total_latency,
            step_count=self.step_count,
        )


class TensorRole(str, Enum):
    """Role of a tensor within a subgraph execution."""
    BOUNDARY_INPUT = "boundary_input"
    BOUNDARY_OUTPUT = "boundary_output"
    EPHEMERAL = "ephemeral"
    RESIDENT = "resident"
