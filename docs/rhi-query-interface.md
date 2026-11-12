# RHI interface queries

RHI objects expose `QueryInterface(InterfaceID, void **)` and
`QueryInterface<T>()`. A successful query transfers one reference. An unsupported
interface returns `false` / `nullptr`; the output pointer is cleared on failure.
Passing a null output argument returns `false`.

```cpp
ObjPtr<IBuffer> buffer = object->QueryInterface<IBuffer>();
if (!buffer)
  return false;
// ObjPtr adopts the query's reference and releases it at scope exit.
```

`QueryInterfacePtr<T>(source)` combines a nullable source, the query, and `ObjPtr`
ownership. Internal code may use its temporary `.Get()` within an expression.
When retaining the resulting raw pointer afterward, the original object must
remain alive, just as with the previous borrowed casts. RHIX containers forward
queries to the underlying RHI object; their ordinary `Get()` remains borrowed.

## Shared backend interface audit

Common backend interfaces are retained only for actual shared consumers or a
single API-independent service implementation. Matching method names across
Vulkan, D3D12, and Metal are insufficient justification for another interface.

Of the original 16 `IBackend*` interfaces, 12 have been removed:

| Removed interface | Current home |
| --- | --- |
| Program, ComputeProgram, RayTracingProgram | Native ProgramBase owns the typed device query |
| Sampler | Native sampler; descriptor validation remains a free shared function |
| BindingGroup, BindingVersion | Native binding group/version; CurrentVersion returns the native type directly |
| DownloadToken | Native download token; CopyData is private to the final implementation |
| Device | Native device; collector wait helpers are private implementation methods |
| Queue, CommandContext | Native submission protocols; EndRecording is private implementation detail |
| Timeline | Native timeline with typed device ownership |
| Image | Native image, including queue eligibility and storage support |

The remaining four interfaces have concrete shared responsibilities:

- `IBackendBuffer::DeviceIdentity`: `ValidateGeometryBuffers` checks ownership
  without knowing the API. Native buffer interfaces separately expose typed Owner.
- `IBackendWindow::GetImGuiContext`: the shared GLFW registry and callbacks need
  this capability. Frame acquisition, presentation, queue identity, and ImGui
  recording now belong to native window interfaces.
- `IBackendSubmissionToken`: the producer facet of the single shared
  `SubmissionTokenImpl`, paired with frontend `ISubmissionToken`. Queues complete
  the same API-independent status object; they do not access its implementation.
- `IBackendShaderBindingTable`: internal regions of the single shared
  `ShaderBindingTableImpl`, paired with frontend `IShaderBindingTable`. Keeping
  this service interface avoids duplicating the shared object or exposing its
  concrete storage to native consumers.

Mock queue/device protocols live only in the test target. Native tests dispatch
to actual backend interfaces, rather than requiring production common facets
solely for test inspection. All native interfaces remain stateless, single-chain
interfaces; only final implementations instantiate Object and own state.

### Cleanup validation

At `6d119ff7`, Windows Release built successfully. With window tests enabled,
220 tests passed and two were skipped. Another 14 ray-tracing tests passed;
the two known DXR compacted-BLAS update cases were excluded. The library has
90 distinct registered interface UUIDs, and pre-commit passed.

On Meshy-PC-3090, GCC and Clang each checked 92 RHI/test translation units from
an isolated source snapshot with `-fsyntax-only -fno-rtti`; both reported zero
failures. This is compiler validation, not a Linux runtime test. Metal sources
were migrated and reviewed, but this cleanup has not been compiled on macOS.

## Identification and implementation

`InterfaceID` contains two `uint64_t` values. `ParseInterfaceID` accepts the
canonical 36-character UUID spelling, with either case for hexadecimal digits,
and returns an empty optional for malformed input. The halves follow textual
order, independent of host endianness. They are not the Windows GUID memory layout.

`rhi_util.h` defines the ID, parser, UUID templates/macros, and public interface
forward declarations. `backend_util.h` declares only shared backend interfaces;
native backend declarations belong to `vulkan_util.h`, `d3d12_util.h`, and
`metal_util.h`. Each interface header defines its `uuid<T>()` specialization with
`LONGMARCH_RHI_INTERFACE(Type, Parent, Text)` at global namespace scope. The specialization
is inline, and `InterfaceTraits<Type>::Base` records the direct parent (`void`
for the root IObject). Including the header in multiple translation units does not create
multiple-definition errors. No separate declaration macro or `.cpp` registration
is needed. Parsing remains constexpr and invalid UUID literals fail at compile
time. Unregistered
types cannot instantiate it. There is no runtime registration map, allocation,
RTTI, or compiler-specific UUID extension. The 90 current library IDs are unique UUIDv4 values. Surviving interfaces
retain their IDs; removed interfaces have no registrations.

Every inheritable RHI object layer is a data-free interface. Virtual contracts
remain pure. The final implementation directly declares its members and
implements operations on them. There are no interface backing-state structs or
virtual state accessors. Similar implementations may repeat straightforward
code; they do not introduce a state indirection to avoid that repetition.
Stateless algorithms may remain ordinary helpers with explicit inputs.

This applies to acceleration structures, images, programs and native layouts,
binding groups and versions, samplers, downloads, timelines, devices, queues,
command contexts, and windows. The former 22 backing-state headers are removed.
Queue submission records and resource transition records still represent actual
work/data. Metal completion callbacks retain a separate `CompletionState` for
their asynchronous lifetime; it is private to the final timeline implementation,
not a backing store accessed through an interface.

Interfaces use single inheritance. Frontend and native capabilities evolve on
separate chains, joined only by a final implementation:

```cpp
struct IBuffer : IObject { /* frontend contract */ };
struct IBackendBuffer : IObject { /* owner capability */ };
struct ID3D12Buffer : IBackendBuffer { /* native resource */ };
struct ID3D12AllocatedBuffer : ID3D12Buffer { /* host access */ };

class D3D12AllocatedBufferImpl final
    : public Object<IBuffer, ID3D12AllocatedBuffer> {
  // Owns data and implements business methods; Object supplies interface queries.
};
```

There are no aggregate interfaces joining those chains. Backend interfaces do
not inherit frontend interfaces or redeclare their methods. Cross-capability
access uses `QueryInterface`; public operations use an existing public pointer
when one is available. Native interfaces remain only where they add internal
capabilities. Factories formerly housed in empty interfaces are free functions;
public-facing factories return the relevant public interface.

`Object<Primary, Interfaces...>` introduces exactly one atomic reference count
and final `AddRef`/`Release` implementations. Each template argument must have
one unambiguous path to `IObject`; interface diamonds are rejected at compile
time. The template implements final `QueryInterface` automatically. It checks
registered ancestors before the interface itself, visiting chains in template
argument order. Interfaces remain data-free and pure virtual; final classes no
longer declare query maps or repeat ancestors. Registered parents are checked
for a valid unambiguous conversion when the chain is instantiated. Querying
`IObject` uses the Primary chain, regardless of the entry interface. A successful
query adds exactly one reference; failed queries clear the output. Implementation
classes are never inherited.

Objects communicate through hidden interfaces rather than implementation
layouts. Final definitions remain in their `.cpp` or internal `*_impl.h` files.

No additional type tag, query table, cached interface pointer, or second
reference count is stored per object. UUID comparisons and pointer adjustments
are explicit in the query implementation. Numeric conversions, opaque API
handles, byte-buffer conversions, and those pointer adjustments are not
interface queries.

## Lifetime boundaries

The caller still releases child references before releasing the parent Device.
Queue timelines now use the same reference-counted object implementation, but
Queue accessors and Events remain borrowed. Querying a timeline returns an owned
reference, which must also be released before its Device.

Internal resource cleanup must not query an owner that is already being
destroyed: a query would attempt to acquire a new reference after final release.
Buffer implementations and binding versions therefore store their single owner
pointer at the native layer with its backend interface type. Shared code accesses it
through the owner interface where needed. Vulkan program cleanup reuses its
existing native layout-device handle. This avoids both reference resurrection
and duplicate owner caches.

## Performance comparison

`scripts/compare_rhi_offscreen.py` compares two separately preserved Release
`demo_rhix_hello` executables. It alternates their order within paired runs,
shuffles case order, records every sample, and reports geometric mean wall-time
ratios with paired bootstrap intervals. Optional baseline-versus-baseline runs
measure ordinary run-to-run variation. Zero control pairs can be used when
reusing separately recorded controls for the identical baseline binary.

The demo measures rendering after 60 warmup frames, limits the pipeline to three
in-flight frames, and drains the GPU before ending its timing region. Process
startup and shader creation are outside that region. Image readback comparison
is performed separately and must not be enabled for timed runs.

These are end-to-end frame-throughput measurements of the entire migration,
including query reference operations and failure checks. They do not isolate
GPU execution time or the cost of UUID comparisons. Results apply to the tested
scenes, hardware, driver, and build configuration.

### Current implementation stress measurement

Main `e233d265` was compared with RHI `3a1fbc02`, using the identical
`demo/rhi_stress` source from `195fdf3c` in both builds. Both were built in the
same Release build directory with MSVC 14.51.36231, Ninja, `/O2 /Ob2 /DNDEBUG`,
and identical runtime DLLs. Hardware: native Windows, Core Ultra 7 270K Plus,
RTX 5080, NVIDIA 617.42. Validation was disabled for timing.

The draw workload renders 4096 independently bound textured quads at 1280x720,
with 16 submissions and 1 MiB of constant uploads per frame. The shader workload
renders 256 quads at 3840x2160, with four submissions and 2048 dependent fragment
iterations. Increasing iterations from 256 to 2048 increased wall time by
7.10-7.36x in both revisions, confirming substantial shader load.

| Backend / workload | Main ms/frame | PR ms/frame | Paired time change (95% CI) |
| --- | ---: | ---: | ---: |
| Vulkan / draw | 0.867 | 5.561 | +541.55% [+536.59, +546.29] |
| D3D12 / draw | 1.882 | 9.488 | +404.23% [+400.63, +407.69] |
| Vulkan / shader | 8.546 | 8.546 | -0.01% [-0.03, +0.02] |
| D3D12 / shader | 14.583 | 14.610 | +0.18% [-0.05, +0.43] |

Each case has 16 paired comparisons in two independently randomized eight-pair
blocks, plus four interleaved A/A controls. Execution order is balanced per block;
120 warmup frames and three in-flight frame slots are used. Measured frame counts
are 1600/600 for Vulkan draw/shader and 900/350 for D3D12 draw/shader. Full-length
priming runs precede measurement. No samples were removed. Frame times are
arithmetic means; changes use geometric paired ratios with 20,000 bootstrap
resamples stratified by block. Raw data also contains Bonferroni-adjusted 98.75%
intervals for the four wall-time cases. These quantify within-session sampling
variation, not systematic hardware or scheduling effects.

The table restricts both binaries to the eight performance cores. The two blocks
independently reproduce draw-time ratios of 6.412/6.419 for Vulkan and
5.032/5.052 for D3D12. A/A draw variations range from -3.66% to +1.05% and
-4.37% to +1.87%, respectively, far smaller than the regressions. A separate
normal-scheduling check (two balanced pairs per API, all 24 logical CPUs) also
reproduces the regression: Vulkan 9.31-9.47x and D3D12 6.84-6.89x. Those extra
pairs are a sensitivity check, not part of the table or its intervals. The exact
slowdown is scheduling-dependent; the regression is not an affinity artifact.

Both shader cases fall within the predeclared +/-3% practical equivalence band,
including adjusted intervals. This does not mean CPU overhead disappeared:
recording/submission wall time rises from 0.191 to 0.504 ms/frame on Vulkan and
0.405 to 0.984 ms/frame on D3D12, while GPU work dominates total throughput.
This timer includes allocations, synchronization and driver stalls; it does not
isolate UUID lookup, reference counting, or CPU cycles. A subsequent
[profiling and controlled experiment](rhi-draw-profile.md) identifies runtime
UUID string parsing as the dominant regression; forcing constant evaluation
removes 96.2%/92.1% of the excess over main on Vulkan/D3D12. The registration
macro now enforces compile-time parsing with a local constexpr result and retains
its validity static_assert. The draw workload deliberately makes many separate
RHI calls rather than using instancing, so these ratios are not universal costs
for production scenes. No Linux or Metal performance claim is made.

All four main/PR raw output comparisons are byte-identical (eight measured,
two warmup frames); Windows Debug runs of both profiles and both APIs pass.
See [benchmark usage](../demo/rhi_stress/README.md) and
[complete samples, provenance and statistics](benchmarks/rhi-stress-query-interface.json).

### Earlier implementation measurement, 2026-11-11

The following data measures `8c166da4`, before the subsequent interface/state
separation. It must not be treated as performance data for the current design.

Baseline: main `e233d265`; candidate: `8c166da4`. Both use MSVC 14.51.36231,
Ninja Release (`/O2 /Ob2 /DNDEBUG`) and identical runtime DLLs. Machine:
Core Ultra 7 270K Plus, RTX 5080, driver 617.42, native Windows. Normal OS
scheduling and driver clock management were retained; no concurrent builds or
other test workloads were launched during sampling.

Each case rendered 100,000 measured frames at 1280x720 per process, with 12
paired runs in balanced randomized A/B order. Each executable first completed
a full-length priming run for every case. Four preceding A/A pairs per case
used the same preserved baseline executable, verified by SHA256. No samples
from intermediate candidate builds were included. Every valid final sample is
retained in [the raw results](benchmarks/rhi-query-interface.json).

| Backend / scene | Main (us/frame) | QueryInterface (us/frame) | Paired time change (95% CI) | A/A pair range |
| --- | ---: | ---: | ---: | ---: |
| Vulkan / cube | 28.84 | 31.99 | +10.95% [+5.55, +16.43] | -4.37% to +11.06% |
| Vulkan / texture | 30.28 | 29.50 | -2.55% [-4.44, -0.30] | -3.66% to +8.31% |
| Vulkan / ray_query | 97.73 | 97.87 | +0.14% [+0.03, +0.27] | -0.12% to +0.94% |
| D3D12 / cube | 34.53 | 39.80 | +15.24% [+14.25, +16.26] | -2.60% to -0.37% |
| D3D12 / texture | 31.67 | 32.47 | +2.55% [+0.96, +4.07] | -2.35% to +7.02% |
| D3D12 / ray_query | 83.00 | 83.59 | +0.71% [+0.39, +1.11] | -1.32% to +0.05% |

Positive changes mean slower rendering. Frame times are arithmetic means;
changes are geometric means of paired ratios, so they need not equal the ratio
of those arithmetic means. Intervals use 10,000 paired bootstrap resamples,
are per case, and are not adjusted for multiple comparisons. They quantify
sampling variation within this experiment, not possible systematic clock,
driver, scheduling, or session drift. A/A ranges are descriptive observations
from four pairs, not confidence bounds or a proof of an overhead threshold.

The clearest regression is D3D12 cube, with a consistent roughly 15% increase.
Vulkan cube also regressed on average, but varied more between pairs. Small
texture and ray-query changes should not be advertised as reliable gains or
universal overhead estimates, given the observed control variation. In
particular, the Vulkan texture result does not establish an optimization.
This migration therefore cannot be described as performance-neutral.

Correctness is checked separately using exact raw image comparisons; capture
hashes are included with the samples. These measurements do not estimate
Metal performance or Linux performance.

## Internal interface access audit

The follow-up audit covers RHI helpers, all three backends, and RHIX wrappers.
Compared with `a0579640`, `QueryInterfacePtr` call sites fell from 364 to 222
(142 removed). This is a source count, not a measured performance improvement.

- WindowImGui and ImagePresenter retain the borrowed `IDevice` interface they
  actually use. Download tokens retain `ISubmissionToken` directly.
- Native queues, contexts, windows, ray-tracing programs, and Vulkan/Metal core
  owners store the appropriate backend interface. Native queue lookup no longer
  goes through a public queue and back. Device feature checks use physical-device
  information already available to the backend.
- Graphics recording uses the original frontend program/image arguments for
  descriptions and validation. Vulkan framebuffer creation receives its extent;
  image barriers obtain the native aspect mask directly. Buffer copies share a
  private native-interface helper after entry-point validation, avoiding the
  upload/download lease-to-frontend-to-native roundtrip.
- Descriptor writes reuse interfaces/handles obtained during validation.
  Collector iterations reuse their timeline query. Scratch pools track their
  allocation capacity; temporal leases derive capacity from their bucket level.
- GLFW registration receives both interfaces from the final window object.
  Lookup retains the frontend interface for the callback, keeping both borrowed
  facets alive without querying during input dispatch or introducing a cycle.
- RHIX `Get()` recovers the exact interface subobject supplied to its typed
  constructor. The container preserves that subobject rather than canonicalizing
  it to `IObject`. This uses a defined static downcast along the same frontend
  inheritance path, never a cast between frontend and backend facets. A test
  covers a noncanonical buffer interface, copy/move, release, and explicit QI.

Queries remain at capability boundaries: external resources/events need backend
and ownership validation; heterogeneous binding references need the requested
native resource capability; factories expose an owned frontend interface from
native storage; submission completion and frontend observation use independent
facets; resource retention requests canonical object identity. RHIX acceleration
structure subtype access still queries because the assignable base wrapper can
hold a different subtype. These are not replaced with assumed cross-facet casts.
Caching raw acceleration-structure handles across recordings is also avoided:
the backing allocation can change during build/compaction.

## Validation

The following checks apply to the implementation with parallel interface chains.
The historical lightweight measurements apply only to the older implementation;
the stress measurements explicitly identify the current RHI revision above.

- Windows Release: 220 passed, two Debug-only queue-diagnostic tests skipped
  (`--gtest_filter=-*RayTracing*`, `LONGMARCH_RHI_WINDOW_TESTS=1`).
- Windows Debug: 48
  interface, timeline, offscreen, window, ImGui, and scratch-pool tests passed.
- Main versus final Release: ten raw color/depth outputs are byte-identical
  across Vulkan/D3D12 cube, texture, ray_query, and raytracing demos (nine frames
  per capture run). The full ray-tracing regression suite was not rerun because
  of the previously identified DXR compressed-BLAS validation issue; both
  backends' ray-tracing pipeline demos were explicitly exercised instead.
- Meshy-PC-3090: GCC Release build passed; isolated regression runs yielded
  105 passed, nine skipped, and no failures. Each test runs in a fresh process
  to avoid the previously observed repeated-Vulkan-load/static-TLS problem.
- Clang syntax checks passed for 88 RHI/test translation units. Six interface
  tests executed successfully with both GCC and Clang with RTTI disabled.
- Historical audit before the shared-backend cleanup: 93 UUID registrations were unique.
  The current count and interface retention rationale are documented above.
  RHI contains no `dynamic_cast` or virtual inheritance. All 22 former state
  wrappers/accessors are absent. The audit checked 88 non-root named interfaces
  plus four nested lease interfaces and 63 final `Object<...>` implementations.
  All interface chains are single-inheritance and frontend/backend chains are
  separate. Registered parent chains cover the full interface ancestry without duplicate
  inheritance paths (except the intentional independent `IObject` roots). The IDs are
  unchanged. Members and implementation methods belong to final classes.
- Metal source was migrated, but no macOS build or runtime validation was
  available. Windows/Linux checks do not substitute for that validation.
