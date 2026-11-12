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
remain alive, just as with the previous borrowed casts. `ObjPtr<T>::Get()` returns a borrowed pointer.

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
RTTI, or compiler-specific UUID extension. Interface IDs are unique UUIDv4 values. Surviving interfaces
retain their IDs; removed interfaces have no registrations.

Every inheritable RHI object layer is a data-free interface. Virtual contracts
remain pure. The final implementation directly declares its members and
implements operations on them. There are no interface backing-state structs or
virtual state accessors. Similar implementations may repeat straightforward
code; they do not introduce a state indirection to avoid that repetition.
Stateless algorithms may remain ordinary helpers with explicit inputs.

This applies to acceleration structures, images, programs and native layouts,
binding groups and versions, samplers, downloads, timelines, devices, queues,
command contexts, and windows.
Queue submission records and resource transition records still represent actual
work/data. Metal completion callbacks retain a separate `CompletionState` for
their asynchronous lifetime; it is private to the final timeline implementation,
not a backing store accessed through an interface.

Interfaces use single inheritance. Frontend and native capabilities evolve on
separate chains, joined only by a final implementation:

```cpp
struct IBuffer : IObject { /* frontend contract */ };
struct ID3D12Buffer : IObject { /* native resource */ };
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
layouts. Interface and final implementation declarations share the corresponding
backend header; `*_impl.h` declarations are merged into those headers.
Classes defined entirely inside a `.cpp` remain local to that implementation.

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

## Shared backend capabilities

Frontend and native interfaces evolve on independent single-inheritance chains.
Only three common backend capabilities remain because they have shared consumers:

- `IBackendWindow::GetImGuiContext` serves the shared GLFW callbacks.
- `IBackendSubmissionToken` publishes the result of the shared submission token.
- `IBackendShaderBindingTable` exposes the regions of the shared SBT implementation.

Resource, queue, device and command-context consumers use their native backend
interfaces. Same-device buffer use is a caller contract asserted only in Debug.
Internal code stores the frontend/native interface it actually needs and reuses
known interfaces rather than querying repeatedly. Resource retention uses
canonical `IObject` identity so two facets of one object share one entry.

Window teardown unregisters callbacks, waits for the owning queue under its
submission mutex, releases the mutex, releases presentation resources, and then
destroys ImGui and the native window. Vulkan retains the separate acquisition
fence waits; D3D12 releases its swapchain before the HWND. TLAS native storage is
destroyed while its BLAS references still exist. No resource destructor adds a
global device wait.

Magnify event storage and Cocoa monitor ownership belong to final backend window
classes. Dispatch retains the frontend window throughout callbacks; native
monitor teardown precedes GLFW destruction. Non-Apple windows report no native
magnify support.

## Regression coverage

Keep tests of actual production behavior: UUID parsing and recursive queries,
canonical identity and reference ownership, typed interface owners, descriptor
versioning, resource lifetimes, GPU timeline/collector progress, swapchain texture
rejection, and gesture callback lifetime/isolation. Metal regressions additionally
cover frontend/native retention and active-render-pass validation.

The former command-context mock suite copied Queue, Collector and Window
algorithms into the test executable after their shared production bases were
removed. Those tests no longer exercised native submission/retirement behavior
and have been removed, along with their test-only interface hierarchy. Core
selection tests retain a small frontend-only device stub and still exercise the
production selection helper. Native tests remain; deterministic fault injection
for blocked next-frame preparation and post-submit allocation failure is not
claimed to be covered by them.

## Performance and deferred work

The synthetic stress demo and its paired-measurement and summary scripts have
been removed. Historical diagnosis, benchmark source and raw reports remain
available through Git history and fixed-revision links in
[Issue #115](https://github.com/LazyJazzDev/LongMarch/issues/115).
The one-off offscreen comparison script is also removed from the source
tree. Future diagnostic runners and raw reports belong in ignored temporary
directories such as `out/`.
The runtime UUID parsing regression described there has been fixed by forcing
constant evaluation; those pre-fix timings do not describe current performance.
[Issue #116](https://github.com/LazyJazzDev/LongMarch/issues/116) tracks shared
backend workflow extraction. Historical reports and per-run JSON dumps are not
part of the maintained source tree.

## 2048 and Game of Life performance against updated main

Measured on Windows with Intel Core Ultra 7 270K Plus, NVIDIA RTX 5080,
driver 617.42, MSVC 14.51.36231 Release (`/O2 /Ob2 /DNDEBUG`). Baseline:
main `6fe5197c`; PR: `90f53ac2`. Demo source is identical between revisions;
both builds use the same C++ feature definitions and identical runtime DLLs.

Each case has eight paired rounds, alternating main/PR execution order, after
one discarded priming round. Each process warms up for two seconds after
initialization and measures 4,000 frames. The original continuous presentation
mode, two submission slots, depth rendering and 3x-per-axis supersampling stay
enabled. 2048 uses 720x960 and a fixed seed of 1, with AI disabled; gol uses
1280x720 and a paused 256x256 grid initialized at density 0.3 with seed 0.
This measures fixed-scene rendering, not AI search or evolving simulation cost.

Timing uses identical temporary steady_clock instrumentation in both demo Run
loops. Full-frame time includes frame-slot waiting, event polling, OnUpdate and
OnRender. The render column measures OnRender, including submission/presentation
and any waits inside them; it is **not** a GPU timestamp. Startup, shader
compilation, final draining and PNG export are excluded. Values below average
the eight per-run means; smaller is better. Positive deltas mean slower PR runs.

| Demo | Backend | main frame (ms) | PR frame (ms) | Change | main OnRender (ms) | PR OnRender (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | vulkan | 0.1161 | 0.1196 | +3.10% | 0.1067 | 0.1104 |
| 2048 | d3d12 | 0.2510 | 0.2646 | +5.43% | 0.2062 | 0.2120 |
| gol | vulkan | 0.3783 | 0.3871 | +2.32% | 0.1583 | 0.1647 |
| gol | d3d12 | 0.4529 | 0.4758 | +5.06% | 0.2042 | 0.2108 |

Run-to-run variation is material for these sub-millisecond frames:

| Demo | Backend | main run-mean range (ms) | PR run-mean range (ms) |
| --- | --- | ---: | ---: |
| 2048 | vulkan | 0.1092-0.1297 | 0.1051-0.1455 |
| 2048 | d3d12 | 0.2237-0.2830 | 0.2272-0.2991 |
| gol | vulkan | 0.3677-0.3833 | 0.3696-0.4244 |
| gol | d3d12 | 0.4176-0.5137 | 0.4364-0.5295 |

The measured aggregate frame overhead is about 2-5%; OnRender increases by
roughly 4-7 microseconds. Paired rounds fluctuate in both directions, so these
numbers do not establish a precise intrinsic QueryInterface overhead or a GPU
regression. The unchanged shaders and matching outputs support functional
parity, not a claim of identical GPU execution time.

All four main/PR PNG pairs are byte-identical. The final gol/D3D12 pair was
completed separately after an earlier process exited successfully without
producing a timing sample; that incomplete run is not counted. All 64 measured
runs have exactly 4,000 samples. Temporary instrumentation and fixed-seed edits
were removed after measurement; no benchmark behavior is added to the demos.

The local logs remain under `out/query-interface/demo-comparison/`; they are
not repository assets. macOS/Metal performance was not measured.
