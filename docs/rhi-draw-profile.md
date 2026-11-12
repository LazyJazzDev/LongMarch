# Draw-heavy RHI profiling

The dominant regression in PR114 is repeated **runtime parsing of UUID string
literals**, not a demonstrated GPU or graphics-driver bottleneck. The interface
registration macro calls a `constexpr` parser from an ordinary runtime expression.
MSVC 14.51.36231 leaves the 36-character parsing loop in optimized code. Forcing
constant evaluation removes most of the measured regression without changing
interface traversal, reference ownership, resource bindings, or GPU commands.

This diagnoses RHI `3a1fbc02` (repository head `fc18ea1b`), against main
`e233d265`, with benchmark source `195fdf3c`. Measurements use Windows Release
`/O2 /Ob2 /DNDEBUG`, Core Ultra 7 270K Plus performance-core affinity, RTX 5080,
and NVIDIA driver 617.42. They do not establish GCC/Clang/Metal behavior.

## Evidence and measurement boundaries

1. Temporary scopes measured command-context methods and `Object::QueryInterface`
   on the recording thread after warmup. They accumulate call counts and inclusive
   and exclusive wall time. Four source files were instrumented; the production
   code is restored after the experiment. Timers change the workload, so their
   timings identify hotspots, not uninstrumented cost percentages.
2. Nsight Systems 2026.3.2 captured D3D12 API/workload events and nested NVTX ranges
   for a separate 10-frame run. It confirms 339,200 QueryInterface ranges and the
   repeated `CmdBindGroup`, `PrepareGroups`, and `PrepareDrawBuffers` paths.
   Nested inclusive range percentages must not be added as a CPU-time breakdown.
   CPU sampling was disabled by Nsight because the process lacks administrator
   privileges. Vulkan API tracing also failed at layer registration; no Vulkan
   API trace or sampled CPU stacks are claimed.
3. A standalone compiler probe uses the production parser and the same registration
   expression, with `/O2 /Ob2 /DNDEBUG /std:c++17 /FAs`. The original expression
   emits character loads, hex classification, shifts and a loop backedge at 36
   characters. The constant-evaluated version emits two 64-bit immediate values
   and stores, with no parsing loop.
4. An uninstrumented experimental binary changes **only** the macro's return
   expression to a local `constexpr` value. Four comparison rounds run main,
   original PR, and this variant, with forward/reverse order balanced. Both APIs
   use the same original draw workload, 120 warmup frames and three frames in
   flight; Vulkan measures 1000 frames/run, D3D12 800. A 300-frame priming run
   precedes each binary/API combination. No sample is discarded. No sampling,
   NVTX ranges or profiling timers are enabled in these comparisons.
5. The variant's eight-frame/two-warmup raw captures are byte-identical to the
   original PR captures on both backends. Runtime DLLs and benchmark source are
   unchanged. All individual timings and executable hashes are in the
   [profiling data](benchmarks/rhi-draw-profile.json).

## Where the work accumulates

The original PR's instrumented 100-frame run gives these counts:

| Operation | Vulkan calls/frame | D3D12 calls/frame |
| --- | ---: | ---: |
| Object::QueryInterface | 22,160 | 33,920 |
| CmdBindGroup | 4,608 | 4,608 |
| CmdBindVertexBuffer | 4,096 | 4,096 |
| CmdBindIndexBuffer | 4,096 | 4,096 |
| CmdDrawIndexed | 4,096 | 4,096 |

Vulkan's largest enclosing recording scope is `CmdBindGroup`, including native
binding-version lookup, resource lookup and binding. D3D12 also queries native
resource interfaces in `PrepareGroups` and `PrepareDrawBuffers` on each draw.
`RetainResource` requests canonical `IObject` identity even for objects already
retained by the context. Successful queries additionally perform AddRef/Release.

The expensive UUID expression appears both at the caller (`uuid<Requested>()`)
and inside `Object`/ancestor matching (`uuid<Candidate>()`). Timing only the
virtual QueryInterface body misses the caller-side parsing. Query counts also
are not parser counts: an individual query can compare multiple ancestor IDs.

The standalone timer run measured 3.923 ms/frame in QueryInterface bodies on
Vulkan and 6.753 ms/frame on D3D12, with instrumented total frame times of 8.341
and 13.077 ms. These numbers include timer perturbation and must not be substituted
for the uninstrumented comparison below.

## Controlled result

| Backend | Main ms/frame | Original PR ms/frame | Constant UUID ms/frame | Removed excess over main |
| --- | ---: | ---: | ---: | ---: |
| vulkan | 0.861 | 5.463 | 1.035 | 96.2% |
| d3d12 | 1.856 | 9.463 | 2.454 | 92.1% |

These are arithmetic means of four runs, not estimates from the instrumented
trace. The constant variant ranges are 1.010-1.066 ms/frame (Vulkan) and
2.440-2.472 ms/frame (D3D12). It reduces original PR frame time by 81.0% and
74.1%, respectively. Relative to main, it leaves 20.2% and 32.2% additional
time. The removed-excess fraction is `(PR - variant) / (PR - main)`; it is
not the percentage of CPU samples attributed to parsing. No confidence
interval is claimed from these four diagnostic rounds.

The experimental delta is:

```cpp
// Original: constexpr-capable parser, but runtime evaluation is permitted.
return *ParseInterfaceID(Text);

// Experiment: initialization requires compile-time evaluation.
constexpr auto id = *ParseInterfaceID(Text);
return id;
```

The original static_assert validating the UUID remains. Merely marking the
parser, or even a caller, `constexpr` does not require constant evaluation at an
ordinary runtime call site. A local `constexpr` value or an inline constexpr
trait member provides that guarantee.

## Possible changes, in priority order

1. **Force UUID constant evaluation in the registration macro.** This is the
   smallest change, preserves the accepted public API and recursive lookup, and
   is the only proposed change measured here. Prefer one constexpr parsed value,
   a static_assert on its validity, and returning that value. Check optimized
   generated code as well as functional tests; another throughput comparison
   should accompany the actual fix.
2. **Keep the consumed native resource interfaces in immutable binding versions.**
   Resolve and validate a buffer/image interface when the binding is written,
   then reuse it when recording. An owning reference or a clearly defined owner
   must keep borrowed native pointers alive. Backend-specific derived capabilities
   must remain validated; do not downcast between parallel interface chains.
   This can reduce both lookup and atomic reference traffic. Its benefit is not
   quantified by the constant-evaluation experiment.
3. **Reuse conversions and canonical identity within a command context.** Retain
   a resource once and reuse its native pointer and canonical identity for later
   binds. Cache entries must be bounded by context lifetime and cannot survive
   resource destruction. Public QueryInterface must still AddRef on success.
   This targets repeated QI/AddRef/Release rather than changing their contract.
4. **Track dirty state for repeated draw preparation.** D3D12 currently revisits
   bound resources in `PrepareGroups` and `PrepareDrawBuffers` for each draw.
   Cache immutable layout/descriptor information and skip only demonstrably
   redundant work. Resource-state changes, UAV ordering, render-target feedback
   checks, heap rollover and binding-version changes still require correct
   invalidation. This is a more invasive follow-up and needs separate tests and
   profiling; it is not necessary to fix UUID parsing.

Instancing or indirect draws could reduce application call volume, but would
change this benchmark's purpose and would not repair the framework regression.
The macro-only experiment leaves a smaller residual difference versus main;
it does not prove that reference counting alone accounts for that remainder.

The profiling experiment originally reverted both instrumentation and its macro
change. The subsequent fix now uses a local constexpr parsed result in the
production registration macro, validates it with static_assert, and returns the
contained ID. Temporary instrumentation remains absent. Nine existing interface
and ownership tests pass. The retained experimental executable and Nsight
reports are under `out/query-interface/profile/` for local inspection.


Post-fix Release verification repeats two balanced old/fixed pairs per backend,
800 measured frames and 120 warmup frames, with performance-core affinity:

- vulkan: 5.525 -> 1.014 ms/frame.
- d3d12: 9.515 -> 2.436 ms/frame.

These are short fix-validation runs, separate from the four-round experiment
above. Both backend captures remain byte-identical to the original PR.
