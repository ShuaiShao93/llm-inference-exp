---
name: optimize-llm
description: Quantify compute and DRAM utilization at both the kernel level and end-to-end, reconcile the two, and turn the gap into a ranked list of optimization targets. Use when asked how well a workload uses the GPU, whether a kernel is worth optimizing, why a fast kernel doesn't make the model fast, or where the headroom is.
argument-hint: [--cmd <benchmark cmd>] [--kernel <regex>] [--bytes <modelled GB>]
allowed-tools: [Bash, Read, Write, Edit]
---

# Optimize LLM Inference: Utilization Accounting

`profile-llm` tells you *where* time goes. This skill tells you *whether that time is
justified* — by measuring achieved compute and DRAM throughput against hardware peak at
two levels, then reconciling them. **The discrepancy between the two levels is the most
informative single number in the whole exercise**: it localizes waste that neither level
shows alone.

## Arguments

$ARGUMENTS

Parse optional args: `--cmd` (benchmark command), `--kernel` (regex for the kernel of
interest), `--bytes` (your modelled traffic in GB, if you already have one). Infer from
context when absent.

## The core idea

Three numbers per level:

| | formula |
|---|---|
| DRAM utilization | `bytes_moved / time / peak_bandwidth` |
| Compute utilization | `FLOPs / time / peak_FLOPS` (at the relevant precision) |
| Arithmetic intensity | `FLOPs / bytes_moved` |

Compare AI against **machine balance** = `peak_FLOPS / peak_bandwidth`. Below balance the
work is memory-bound and only byte reduction helps; above it, only FLOP reduction helps.
Machine balance is the single most useful hardware constant to know — compute it once per
GPU and keep it handy.

---

## Step 0: Establish the hardware peaks

Never take peak bandwidth from a spec sheet alone — ECC, clock throttling, and refresh
overhead put achievable peak below theoretical. Two ways, use both:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.memory --format=csv
```

**Preferred: let ncu tell you.** ncu's `dram__throughput.avg.pct_of_peak_sustained_elapsed`
already divides by the peak *it* measured on this silicon. Reading that metric is more
trustworthy than hand-dividing by a number you looked up. Cross-check once: confirm
`dram__bytes.sum / gpu__time_duration.sum` divided by your assumed peak reproduces ncu's
percentage. If it doesn't, your assumed peak is wrong — trust ncu's.

Record peak FLOPS at the precision actually used (BF16 vs FP8 vs FP4 differ by 2× per
step on tensor cores, and sparsity numbers in marketing material are usually 2× inflated).

---

## Step 1: The two-number ratio (do this first)

When the question is *what does feature X cost* (LoRA, speculative decode, a new kernel),
it reduces to two numbers. Get these before touching ncu — they bound what any kernel-level
win can buy, and they are often the whole answer.

```
theoretical_ms = modelled_bytes / peak_bandwidth      # from the shapes
actual_ms      = time(X on) - time(X off)             # A/B the same cell
utilization    = theoretical_ms / actual_ms
```

Read it directly: **>80% → bandwidth-bound**, so only moving fewer bytes helps and tiling
work is spent. **<50% →** something other than bandwidth dominates; go find it before
refining the byte model.

Two rules that decide whether the ratio means anything:

- **The denominator is the A/B delta, not the sum of X's own kernel durations.** They
  differ, because adding X perturbs the kernels around it — cache pressure, occupancy,
  scheduling — so summing only X's kernels credits it for time it didn't add. Observed
  skew from this substitution: several points of utilization, always in the flattering
  direction, which understates the slack left on the table.
- **Model bytes from the real per-layer shapes, never from a uniform reading of the
  config.** Stacks are routinely heterogeneous (per-layer FFN widths, periodic full-
  attention layers, merged projections with unequal slices), and frameworks often wrap more
  modules per layer than the obvious q/k/v/o/gate/up/down set. Read widths off the actual
  weight tensors. Observed error from a uniform model on a heterogeneous stack: 1.7× too
  few bytes, which reads out as tens of points of missing utilization and invites a hunt
  for time that was never lost.

For a whole model rather than one feature, the same ratio applies with `actual_ms` = wall
ms/run, plus FLOPs to place it against machine balance: `2 * params * tokens` for the linear
layers and `~4 * layers * tokens² * hidden` for attention (the quadratic term dominates at
long context). Also note GPU-busy ms/run from nsys — wall minus GPU-busy is launch gap or
CPU-bound time, worth knowing before optimizing any kernel.

**Write down that `modelled_bytes` is modelled** — Step 3 tests it.

---

## Step 2: Kernel-level utilization with ncu

Target narrowly; ncu costs ~100× per profiled launch.

```bash
ncu --kernel-name regex:"<kernel>" --launch-count 3 \
    --metrics dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
launch__grid_size,\
launch__occupancy_limit_registers \
    --csv -o - <benchmark cmd>
```

Notes that repeatedly matter:

- **`sm__throughput` understates tensor-core work.** It's a composite of all pipes; a
  GEMM saturating the tensor pipe can show a modest `sm__throughput`. Always read the
  `sm__pipe_tensor_op_*` metric alongside it before calling a GEMM "not compute-bound."
- **Counter permission**: on `ERR_NVGPUCTRPERM`, prefer `sudo -E $(which ncu) ...` over the
  documented modprobe reload — it needs no kernel-module unload, so it is safe on a live
  host with someone else's work on it. Reload only if sudo is unavailable.
- **Parse ncu CSV by seeking the line starting with `"ID"`.** Any stdout the target program
  prints lands above the header and will otherwise be consumed as the header.
- Profile in the **timed** phase: `--launch-skip N` past warmup, or gate on
  `cudaProfilerStart`.

---

## Step 3: Reconcile the two levels — the actual diagnostic

Put kernel-level and end-to-end utilization side by side. Interpret the gap:

| Observation | Meaning | Next probe |
|---|---|---|
| kernel ≈ e2e | Model is sound; utilization is what it is | Go to Step 4 |
| **kernel ≫ e2e** | Either untracked bytes or untracked time | Below |
| kernel ≪ e2e | Almost always a bookkeeping error — you profiled an unrepresentative launch, or double-counted bytes | Re-check shapes |

When kernel utilization is high but e2e is low, decompose:

```
wall_delta = Σ(launches_i × duration_i)  +  gap
real_bytes = Σ(launches_i × dram_bytes_i)
```

- Get `launches_i` and `duration_i` from **nsys**, bucketed by grid size — grid size is a
  free shape fingerprint, so distinct layer shapes separate into distinct buckets without
  any source reading.
- Get `dram_bytes_i` from **ncu**, one representative launch per bucket.
- If `Σ(launches × duration)` ≈ wall delta → time is all inside the kernels, so the gap is
  **untracked bytes**: your byte model is too low. Usual causes are in the traps below.
- If it falls well short → the gap is **untracked time**: launch overhead, memsets,
  synchronization, or a kernel your classifier missed.

**Get launch counts by measurement, not by reading the source.** Whether a framework fuses
N projections into one call or issues N calls changes both the byte model and the launch
count, and the answer is often not what the code appears to say.

### Split the remaining gap into bytes vs inefficiency

Once ncu has given you real bytes, the Step 1 gap separates into two causes with different
fixes:

```
gap_ms          = actual_ms - theoretical_ms
excess_bytes_ms = (measured_bytes - modelled_bytes) / peak_bandwidth   # traffic you missed
inefficiency_ms = gap_ms - excess_bytes_ms                             # tiling / occupancy
```

`measured_bytes > modelled_bytes` means the kernels move traffic your model didn't predict —
usually a read-modify-write missing L2, or partial-tile waste. That is fixed by changing
what the kernel touches, not by tuning tiles. Only `inefficiency_ms` is addressable by
tiling.

**Then check utilization per shape before believing the aggregate.** A weighted average
hides concentration: it is common for one shape to hold nearly all the slack while every
other shape is already pinned at 90%+. Quoting the average as though the slack were spread
evenly leads to tuning kernels that have nothing left to give. Rank shapes by
`traffic_share × (ceiling - measured_utilization)` and expect the list to be very top-heavy.
Use ~90% of peak as the practical ceiling for a streaming kernel, never 100%.

---

## Step 4: Verdict and what it licenses

| DRAM % | Compute % | Verdict | Only these help |
|---|---|---|---|
| >80 | any | **Bandwidth-bound** | Move fewer bytes: lower precision, fuse to avoid round-trips, remove redundant reads, avoid read-modify-write |
| <60 | >70 (tensor pipe) | **Compute-bound** | Fewer FLOPs: lower precision, better algorithm, sparsity |
| <60 | <60 | **Latency-bound** | Bigger work per launch, fix occupancy, more ILP, CUDA graphs |
| >80 | >70 | Balanced — near roofline corner | Structural change only |

**A kernel above ~90% of DRAM peak has no tiling win left.** Bytes convert 1:1 into time,
so any speedup must come from moving fewer bytes. Conversely, if a kernel is at 40% of peak
with low compute, the win is in launch/occupancy/tiling and is often large and cheap.

### Ranking targets

Rank by **traffic share × recoverable fraction**, not by how bad a kernel looks:

- A kernel at 95% of peak that is 70% of total traffic has little recoverable headroom.
- A kernel at 60% of peak that is 5% of traffic is irrelevant no matter how bad it looks.
- Compute each candidate's **structural floor** (the bytes an ideal implementation must
  move) and compare to measured. Measured ÷ floor is the honest upper bound on the win.
  Quoting a microbenchmark speedup without this is how a 5% win gets sold as 3×.

---

## Traps that have actually produced wrong conclusions here

- **Modelled traffic that assumes each operand is read once.** If the kernel's grid
  replicates a dimension, the shared operand is re-read once per grid slot. This produced a
  3× error and a false "2.4× headroom" claim. Always confirm a modelled byte count against
  `dram__bytes.sum` before acting on it.
- **Broadcast/reduction dimensions placed in the grid.** A GEMM whose N is tiny (e.g. a
  low-rank projection) may re-read a large shared input per slice rather than hold small
  accumulators. Grid choices tuned for small batches invert at prefill-scale M.
- **`split_k > 1` multiplies output write traffic** and forces an output memset. Cheap when
  the output is large relative to reads, expensive when the output is a sliver.
- **High L2 hit rate can mask redundant DRAM reads** — and lowering L2 hit rate can still be
  a net win if it removes more traffic elsewhere. Read L2 and DRAM together, never alone.
- **One-shape microbenchmarks don't generalize.** Weight every shape by its measured share
  of total traffic before quoting an aggregate win. The dominant shape is frequently not
  the one that's easiest to benchmark.
- **Uniform-slice assumptions.** Merged projections are often unequal (GQA q/k/v are not
  three equal widths). Modelling them as uniform misstates output width substantially.
- **Frameworks run work for entities whose weights are zero or absent.** Wrapping is
  typically decided once at init from the base model's inventory, while "is this actually
  used" is only known per request. The usual result is full-cost kernels computing a no-op.
  Count what the framework *launches*, not what the config *targets* — and when the counts
  disagree, the difference is free to reclaim.
- **Framework default tile/config heuristics are usually tuned for decode**, where batch is
  small. At long-context prefill the same heuristic can be far off. Check whether the
  framework exposes a tuned-config override before patching source; many do, and a
  config-only A/B is both faster and more publishable. **Only actually tune when asked
  to** — a tuned number describes a config the framework does not ship, so every reported
  figure stays on defaults unless the request was explicitly about tuning.
- **Two GPUs that agree is not a scaling law — get a third before you name a trend.** Two
  points always fit a line, and if they happen to land near each other it is very tempting to
  read that as "this kernel doesn't scale with bandwidth." A third GPU can be 3× off that
  line and non-monotonic, which retracts the trend *and* every extrapolation built on it. Cost
  the third measurement against the cost of publishing a wrong mechanism.
- **Never put a microbenchmark and an end-to-end A/B in the same table column.** An isolated
  kernel harness and a whole-path delta have different denominators and different confounds;
  side by side they read as one dataset and will quietly anchor a conclusion. Label the method
  per row, and if a microbenchmark contradicts an end-to-end bound, believe the bound —
  arithmetic on the A/B delta ("this kernel must exceed X GB/s to fit in the measured time")
  is the cheapest way to catch a broken harness.
- **A cross-GPU comparison is only as good as its matched config.** Backend, precision, KV
  dtype and any capacity/rank padding knob must be identical, or the spread you attribute to
  hardware is partly config. Before quoting "N% of peak" across cards, re-run one of them with
  another's config on the same host — it is cheap and it isolates one confound cleanly.
- **Partially-specified override configs can hard-fail** if a lookup level lacks a
  nearest-key fallback. Populate every level the model will actually request.

---

## Reporting

Lead with the Step 1 ratio and name its denominator — "X% of peak, against the A/B delta"
is a claim a reader can check; a bare "X% of peak" is not, since the same bytes over a
kernel-duration sum give a different and flattering number.

State for each claim whether it is **measured** or **modelled**, and name the tool. Report
the structural floor next to the measured number so the recoverable fraction is explicit.
If a conclusion rests on a modelled byte count that hasn't been checked against
`dram__bytes.sum`, say so — that is exactly the claim most likely to be wrong.

If this skill's guidance turns out to be incomplete or wrong for the case at hand, ask
whether to update it.
