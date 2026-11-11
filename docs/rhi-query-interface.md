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

## Identification and implementation

`InterfaceID` contains two `uint64_t` values. `ParseInterfaceID` accepts the
canonical 36-character UUID spelling, with either case for hexadecimal digits,
and returns an empty optional for malformed input. The halves follow textual
order, independent of host endianness. They are not the Windows GUID memory layout.

`LONGMARCH_RHI_UUID(Type, Text)` defines an inline constexpr specialization of
`uuid<Type>()`. Use the macro at global namespace scope before the first query
for the type. Invalid literals fail at compile time; unregistered types cannot
instantiate `uuid<T>()`. Registration uses no runtime map, allocation, RTTI, or
compiler-specific UUID extension. All registered library IDs were generated
once with Python `uuid.uuid4()` and must remain stable for the same interface.

`Object<Interface, ...Bases>` supplies one atomic reference count. It exposes the
primary interface, `IObject`, and any explicitly listed ancestor interfaces.
This keeps the common implementation independent of particular resource types.
Concrete backend classes advertise their implementation interfaces through the
query helpers. Secondary interfaces use ordinary inheritance and forward
reference operations to the primary base's counter. Querying `IObject` through
any supported interface returns the same canonical identity pointer.

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
pointer at the native layer with its concrete type. Shared code accesses it
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
