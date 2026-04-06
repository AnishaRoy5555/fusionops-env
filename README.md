# FusionOps

An RL environment for ML computation graph scheduling. Agents schedule operations on a DAG under memory constraints, balancing operator fusion, tiling granularity, and tensor residency to minimize total execution latency.

The problem is drawn directly from how ML compilers (XLA, Triton, TVM) schedule workloads on hardware with limited on-chip memory.

## Why This Exists

Every ML model compiles down to a graph of operations (matmuls, pointwise ops) that must execute on hardware with a small, fast scratchpad and a large, slow main memory. The compiler must decide:

- Which ops to fuse into a single kernel (eliminating intermediate memory traffic)
- How to tile the computation (trading off compute efficiency vs. memory footprint)
- Which tensors to keep resident between kernels (avoiding redundant loads)
- When to use split-K accumulation (reducing memory at the cost of more compute passes)

Getting this wrong means 2-10x slower training. There is no existing RL benchmark for this task.

## How It Works

The agent receives a computation graph and hardware spec. Each step, it picks a group of operations to schedule together, chooses a tiling configuration `[w, h, k]`, and decides which output tensors to keep in fast memory. The environment computes the resulting latency using a roofline cost model and updates the state.

The episode ends when all operations are covered.

### Cost Model

Latency per tile = `max(compute_time, memory_transfer_time)`. This is the standard roofline model used in real compiler backends.

- **Compute time**: sum of operation costs, with padding penalties when tiling below native granularity
- **Memory time**: data moved to/from slow memory, divided by bandwidth
- **Fusion benefit**: intermediate tensors between fused ops become ephemeral (zero memory cost)
- **Split-K**: for MatMul, reducing `k` below the full reduction dimension trades more compute passes for lower peak memory

The cost model has been verified against all 5 official examples from the [MLSys 2026 competition spec](https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md), including the tricky cases (recomputation semantics, split-K with chained MatMul, traversal order data reuse).

## Action Format

```
SCHEDULE ops=[0,1] config=[128,128,1] retain=[2]
```

| Field | Description |
|-------|-------------|
| `ops` | Operation IDs to group into one subgraph. Must be connected in the DAG. All predecessors must already be scheduled. |
| `config` | Tiling granularity `[w, h, k]`. Output tiles are `w x h`. For MatMul, `k` controls reduction depth. |
| `retain` | Tensor IDs to keep in fast memory after this subgraph. Everything else is evicted to slow memory. |

**Validation rules:**
- Working set (input slices + output slices + retained tensors) must fit in fast memory, otherwise OOM
- Operations must form a connected subgraph
- All predecessor operations must be scheduled (or included in the current subgraph)

## Observation

Each step the agent sees:

- Hardware spec (fast memory capacity, bandwidth, native granularity)
- Full graph structure with per-op status (scheduled or not)
- Tensor dimensions and current locations (slow memory, fast memory, not yet computed)
- Which operations are ready to schedule (all predecessors done)
- Suggested fusion candidates (connected ready ops that share tensors)
- Schedule history (previous subgraphs, configs, latencies)
- Cumulative latency and step count

## Reward and Scoring

**Per-step reward** combines:
- Latency efficiency signal (lower latency per step = higher reward)
- Fusion bonus (+0.05 per additional op fused beyond the first)
- Completion bonus scaled by final score

**Final score:**

```
score = max(0, (naive_latency - agent_latency) / naive_latency)
```

Where `naive_latency` is the cost of scheduling every op individually with no fusion and no retention. Score 0 means no improvement over naive. Higher is better.

## Tasks

| Task | Ops | Structure | Challenge | Expected Baseline |
|------|-----|-----------|-----------|-------------------|
| `task1_linear` | 6 Pointwise | Linear chain | Basic fusion | 0.5 - 0.7 |
| `task2_diamond` | 6 Pointwise | Diamond + skip connections | Retain vs. recompute tradeoff | 0.3 - 0.5 |
| `task3_matmul` | 3 MatMul + 1 Pointwise | Chained MatMul | Split-K, OOM avoidance, memory management | 0.1 - 0.3 |

**Task 1** rewards agents that understand fusion: grouping the 6-op chain into one subgraph eliminates 5 intermediate memory round-trips.

**Task 2** has tensor T1 consumed by 3 downstream ops. The agent must decide whether to retain T1 in fast memory (saving reloads but consuming capacity) or let it evict and reload later. The optimal strategy depends on the schedule order.

**Task 3** has 128x128 MatMuls where the working set barely fits in 50KB fast memory. Fusing two MatMuls requires split-K (k=32 instead of k=128) to avoid OOM. The agent must reason about memory arithmetic.

## Running Locally

```bash
docker build -t fusionops .
docker run -p 8000:8000 fusionops
```

Or without Docker:

```bash
pip install fastapi uvicorn pydantic
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

**Reset** (start a new episode):
```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "task1_linear"}'
```

**Step** (take an action):
```bash
curl -X POST http://localhost:8000/step/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"command": "SCHEDULE ops=[0,1,2,3,4,5] config=[128,128,1] retain=[]"}'
```

**Tasks** (list available tasks):
```bash
curl http://localhost:8000/tasks
```

WebSocket endpoint available at `/ws` for persistent sessions.

## Baseline Agent

The baseline (`inference.py`) uses an LLM via the OpenAI client. The system prompt teaches three heuristics:

1. Fuse adjacent ops in chains to eliminate intermediate memory transfers
2. Use native granularity unless memory forces smaller tiles
3. Retain tensors that the next subgraph will need

The agent reads observations, asks the LLM for an action in `SCHEDULE ops=... config=... retain=...` format, and the environment parses it. No hardcoded strategies.

## Project Structure

```
fusionops-env/
  inference.py           # Baseline LLM agent (OpenEnv submission format)
  openenv.yaml           # OpenEnv metadata
  Dockerfile             # Container
  fusionops_env.py       # Client library
  server/
    app.py               # FastAPI server (HTTP + WebSocket)
  src/
    models.py            # Data structures, graph loading
    cost_model.py        # Roofline latency computation
    validator.py         # Action validation, OOM checks
    observation.py       # Observation text formatting
    environment.py       # Core environment logic
    tasks.py             # 3 fixed task definitions
  tests/
    test_golden.py       # 12 tests against official MLSys examples
```

## Verification

The cost model passes 12 golden tests reproducing every strategy from the [MLSys problem spec](https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md):

- Example 1: Pointwise fusion (strategies A, B, C)
- Example 2: Large tensor tiling
- Example 3: Spill vs. recompute vs. selective residency
- Example 4: MatMul traversal order data reuse (raster vs. snake)
- Example 5: Chained MatMul with split-K (including OOM detection)

```bash
python tests/test_golden.py
# Results: 12 passed, 0 failed out of 12
```

## Spec Compliance

- OpenEnv v0.2.1 compatible
- Typed observation/action models
- `step()` / `reset()` / `state()` endpoints
- Deterministic: same actions always produce the same latencies and scores
- Dockerfile builds and runs on 2 vCPU / 8 GB RAM
- Baseline inference completes all 3 tasks in under 5 minutes
