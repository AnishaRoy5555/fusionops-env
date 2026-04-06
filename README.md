---
title: FusionOps
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# FusionOps

An RL environment for ML computation graph scheduling. Agents schedule operations on a DAG under memory constraints, balancing operator fusion, tiling granularity, and tensor residency to minimize total execution latency.

The problem is drawn directly from how ML compilers (XLA, Triton, TVM) schedule workloads on hardware with limited on-chip memory.

## Why This Matters

Every ML model compiles down to a graph of operations (matmuls, pointwise ops) that must execute on hardware with a small fast scratchpad and a large slow main memory. The compiler must decide:

- Which ops to fuse into a single kernel (eliminating intermediate memory traffic)
- How to tile the computation (trading compute efficiency for memory footprint)
- Which tensors to keep resident between kernels (avoiding redundant loads)
- When to use split-K accumulation (reducing memory at the cost of more compute passes)

Getting this wrong means 2-10x slower training. This is one of the highest-leverage problems in ML systems engineering, and it has no existing RL benchmark.

## What Makes This Hard

This environment is designed to test capabilities that go beyond pattern matching:

**Long-horizon planning.** A decision in step 1 (which tensor to retain) affects whether step 4 will OOM. Greedy strategies fail.

**Memory vs. compute tradeoffs.** Recomputing an operation can be cheaper than reloading its output from slow memory. The agent must reason about this tradeoff per-tensor.

**Non-local decisions.** A tensor produced early may have a consumer many steps later. The agent must track which tensors will be needed and either retain them, recompute them, or accept the reload cost.

**Hardware constraints.** Working set must fit in fast memory. Choices that look optimal locally (full-K MatMul) cause OOM when combined with fusion. The agent must reason about memory arithmetic.

**Mixed op types.** Pointwise ops fuse cheaply. MatMuls have expensive accumulation phases. Fusing them requires split-K, which creates a new tradeoff space.

## How It Works

The agent receives a computation graph and hardware spec. Each step, it picks a group of operations to schedule together, chooses a tiling configuration `[w, h, k]`, and decides which output tensors to keep in fast memory. The environment computes the resulting latency using a roofline cost model and updates the state.

The episode ends when all operations are covered.

### Cost Model

Latency per tile = `max(compute_time, memory_transfer_time)`. This is the standard roofline model used in production compiler backends.

- **Compute time**: sum of operation costs, with padding penalties when tiling below native granularity
- **Memory time**: data moved to/from slow memory, divided by bandwidth
- **Fusion benefit**: intermediate tensors between fused ops become ephemeral (zero memory cost)
- **Split-K**: for MatMul, reducing `k` below the full reduction dimension trades more compute passes for lower peak memory

The cost model has been verified against 12 golden test cases covering all core scheduling scenarios, including the tricky edge cases (recomputation semantics, split-K with chained MatMul, traversal order data reuse).

## Action Format

```
SCHEDULE ops=[0,1] config=[128,128,1] retain=[2]
```

| Field | Description |
|-------|-------------|
| `ops` | Operation IDs to group into one subgraph. Must be connected in the DAG. All predecessors must already be scheduled. |
| `config` | Tiling granularity `[w, h, k]`. Output tiles are `w x h`. For MatMul, `k` controls reduction depth. |
| `retain` | Tensor IDs to keep in fast memory after this subgraph. Must be outputs of the current subgraph. |

**Validation rules:**
- Working set (input slices + output slices + retained tensors) must fit in fast memory
- Operations must form a connected subgraph
- All predecessor operations must be scheduled (or included in the current subgraph)
- Retained tensors must be outputs of the current subgraph (not earlier ones)

## Observation Structure

The observation is structured into clear sections to help LLM agents reason effectively:

1. **LAST ACTION RESULT** (only if previous action failed): Error type, reason, and a precise fix hint
2. **CURRENT STATE**: Completed ops, ready ops, step counter
3. **MEMORY**: Fast memory contents and capacity usage
4. **GRAPH SUMMARY**: All ops with status, all tensors with locations
5. **VALID ACTION EXAMPLES**: 2-4 syntactically valid actions the agent could take right now
6. **CONSTRAINTS**: Hard rules the action must satisfy
7. **BEST PRACTICES**: General strategy guidance (not task-specific)
8. **PROGRESS**: Latency so far vs. naive baseline

The action examples are generated dynamically from the current state and are guaranteed valid by construction. They never reveal the optimal policy.

## Reward Function

**Per-step reward** combines:
- Small positive baseline (+0.02) for any valid step
- Latency efficiency signal (lower latency = higher reward)
- Fusion bonus (+0.05 per additional op fused beyond the first)
- Completion bonus scaled by final score

**Graduated penalties for invalid actions:**

| Error Type | Penalty | Why |
|-----------|---------|-----|
| OOM | -0.05 | Close to valid, just wrong tile size |
| Parse error | -0.10 | Wrong format |
| Already scheduled | -0.10 | Confused about state |
| Connectivity | -0.15 | Wrong subgraph composition |
| Retention | -0.15 | Wrong understanding of semantics |
| Dependency | -0.20 | Clearly wrong order |

The graduated penalties tell the agent what kind of mistake it made, enabling self-correction within a single rollout.

**Final score:**

```
score = max(0, (naive_latency - agent_latency) / naive_latency)
```

Where `naive_latency` is the cost of scheduling every op individually with no fusion and no retention. Score 0 means no improvement over naive. Higher is better.

## Tasks

| Task | Ops | Naive | Strong Strategy | Theoretical Ceiling | Tests |
|------|-----|-------|----------------|---------------------|-------|
| `task1_linear` | 6 PW | 0.000 | 0.669 | 0.669 | Basic fusion |
| `task2_diamond` | 6 PW | 0.000 | 0.422 | ~0.55 | Retention vs recompute |
| `task3_matmul` | 3 MM + 1 PW | 0.000 | 0.347 | ~0.47 | Split-K, OOM avoidance |
| `task4_multistage` | 3 MM + 5 PW | 0.000 | 0.246 | ~0.30 | Long-horizon, skip connections |

**Task 1** rewards basic fusion: grouping the 6-op chain into one subgraph eliminates 5 intermediate memory round-trips. Solvable in 1 step.

**Task 2** has tensor T1 consumed by 3 downstream ops. The agent must reason about whether to retain T1, recompute it, or accept reloads. Solvable in 4-5 steps with the right strategy.

**Task 3** has 128x128 MatMuls where the working set barely fits in memory. Fusing two MatMuls requires split-K. The agent must reason about memory arithmetic. Solvable in 2-3 steps.

**Task 4** is the expert challenge: 8 ops, 3 MatMul stages, a skip connection from the first MatMul output to op 6 (six steps later), and tight memory. Requires fusing every MatMul with its trailing Pointwise (split-K), reasoning about a long-distance tensor dependency, and choosing whether to retain or reload T8. Solvable in 4-5 steps optimally.

## Running Locally

```bash
docker build -t fusionops .
docker run -p 7860:7860 fusionops
```

Or without Docker:

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Endpoints

**Reset** (start a new episode):
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "task1_linear"}'
```

**Step** (take an action):
```bash
curl -X POST http://localhost:7860/step/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"command": "SCHEDULE ops=[0,1,2,3,4,5] config=[128,128,1] retain=[]"}'
```

**Tasks** (list available tasks):
```bash
curl http://localhost:7860/tasks
```

WebSocket endpoint available at `/ws` for persistent sessions.

## Baseline Agent

The baseline (`inference.py`) uses an LLM via the OpenAI client. The system prompt is intentionally minimal:

> You are an agent interacting with an ML graph scheduling environment. Each observation contains valid action examples, constraints, and best practices. When the previous action failed, read the fix hint and apply it. Goal: minimize total latency by fusing connected ops and reducing memory transfers.

The structured observation does the heavy lifting. The agent reads the VALID ACTION EXAMPLES, picks one that fits the current state, and submits. When it gets a fix hint after a failure, it adjusts on the next step.

## Project Structure

```
fusionops-env/
  inference.py           # Baseline LLM agent (OpenEnv submission format)
  openenv.yaml           # OpenEnv metadata
  Dockerfile             # Container
  pyproject.toml         # Build + dependencies
  requirements.txt       # Pip requirements
  fusionops_env.py       # Client library (Pydantic models)
  server/
    app.py               # FastAPI server (HTTP + WebSocket + main entry point)
  src/
    models.py            # Data structures, graph loading
    cost_model.py        # Roofline latency computation
    validator.py         # Action validation, OOM checks
    observation.py       # Structured observation formatting with action hints
    environment.py       # Core environment with error classification
    tasks.py             # 4 fixed task definitions
  tests/
    test_golden.py       # 12 golden tests covering all cost model scenarios
```

## Verification

The cost model passes 12 golden test cases:

```bash
python tests/test_golden.py
```
# Results: 12 passed, 0 failed out of 12

The tests cover:
- Pointwise fusion with different granularities (3 strategies)
- Large tensor tiling across multiple spatial tiles
- Diamond graphs with spill vs. recompute vs. selective residency
- MatMul traversal order data reuse (raster vs. snake)
- Chained MatMul with split-K accumulation and OOM detection

## Spec Compliance

- OpenEnv v0.2.1 compatible
- Typed Pydantic Observation/Action/Result models
- `step()` / `reset()` / `state()` endpoints
- Deterministic: same actions always produce the same latencies and scores
- Dockerfile builds and runs on 2 vCPU / 8 GB RAM
- Baseline inference completes all 4 tasks in under 5 minutes
- Defensive error handling: inference.py never exits with non-zero status
- Auto-installs missing dependencies at runtime as a fallback

## Design Notes

**Why retention is bounded to current subgraph.** This matches real hardware: when a kernel finishes, the scratchpad is repurposed for the next kernel. Long-term caching requires explicit recomputation in the next subgraph or accepting a reload. This forces agents to think in terms of immediate-next-step optimization, just like real compilers.

**Why split-K matters.** Split-K MatMul reduces peak working set by streaming the K dimension in chunks. This unlocks fusion opportunities that would otherwise OOM. The tradeoff is more accumulation passes (slightly more compute). Real compilers like cuBLAS use split-K extensively for memory-bound workloads.

**Why traversal order matters.** When tiling a MatMul output, adjacent tiles share input data. Snake traversal (zigzag) keeps the LHS warm across rows, reducing reloads. Raster traversal (left-to-right then top-to-bottom) wastes this reuse. This is a real consideration in hand-tuned kernel implementations.
