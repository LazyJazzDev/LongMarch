# RHI CUDA buffer and timeline interoperability

When CMake finds the CUDA Toolkit, `grassland_rhi` publicly enables
`LONGMARCH_CUDA_RUNTIME` and links the CUDA runtime. CUDA interop interfaces are
available under that definition; ordinary RHI use does not require CUDA. Compiling
the kernel integration test additionally requires `LONGMARCH_CUDA_ENABLED` (nvcc).

Vulkan supports opaque FD handles on Linux and opaque Win32 handles on Windows.
D3D12 uses shared committed resources and shared fences. Metal has no CUDA path.

## Device selection

`ICore::GetPhysicalDevices()` and `IDevice::GetProperties()` expose
`DeviceProperties::cuda_device`. `PhysicalDevice` remains an alias for the same
properties type. The value is a process-local CUDA Runtime ordinal, matched by
Vulkan device UUID or D3D12 adapter LUID; it is never inferred from the graphics
adapter's enumeration index. Unmatched devices and builds without CUDA report
`-1`. Linked-node D3D12 adapters are excluded.

A match identifies the GPU; it does not guarantee that every external resource
configuration is supported. Creation returns null on unsupported or failed
interop. Select the matching CUDA device before launching kernels or creating
streams, and use streams from that device/context. Internal CUDA API operations
select the object's device and restore the caller's previous device.

## Fixed shared buffer

`CreateCUDABuffer(BufferDesc)` returns an `ICUDABuffer`, also usable as `IBuffer`.
Creation allocates dedicated exportable graphics memory, imports it into CUDA and
maps it once. `GetDevicePointer()` only returns the cached pointer. Both APIs
access the same storage; the pointer remains unchanged for the buffer's lifetime.
There is no resize or per-frame map/unmap.

The buffer starts on the RHI side. Record `CmdReleaseCUDABuffer()` before giving
it to CUDA, and `CmdAcquireCUDABuffer()` before using CUDA's result in RHI. These
commands must be outside a rendering scope. They encode Vulkan external queue
family ownership transfers and D3D12 COMMON-state handoffs. Pair them with the
execution dependencies below. Do not use a buffer on one side after releasing it
until that side reacquires it. Release ends its use in that command context.

The caller prevents conflicting accesses and completes all CUDA use before
releasing the last buffer reference. Accepted RHI submissions retain their
resources until GPU completion. Destruction frees the CUDA mapping, destroys the
CUDA external memory import, then releases the native buffer and allocation.

## Explicit shared timeline

`CreateCUDAInteropTimeline()` returns a zero-initialized shared timeline:

- `SignalCUDA(stream, value)` enqueues a signal after earlier stream work.
- `WaitCUDA(stream, value)` enqueues a wait before later stream work.
- `SubmitInfo::cuda_waits` adds timeline/value dependencies to a queue submission.
- `SubmitInfo::cuda_signals` signals timelines when the submission's commands finish.

All of these are GPU operations; they do not wait for GPU completion on the CPU.
`cudaStream_t{}` / `nullptr` means the default stream. No stream mapping or
registration is maintained. Queue dependency lists are empty by default.

```cpp
using namespace grassland::rhi;
ObjPtr<ICUDABuffer> buffer = device->CreateCUDABuffer({byte_size});
ObjPtr<ICUDAInteropTimeline> timeline = device->CreateCUDAInteropTimeline();
// Check creation and every command/operation result in application code.

ObjPtr<ICommandContext> release = device->CreateCommandContext();
release->CmdReleaseCUDABuffer(buffer.Get());
SubmitInfo before;
before.cuda_signals = {{timeline.Get(), 1}};
ObjPtr<ISubmissionToken> released = queue->Submit(release.Get(), before);

timeline->WaitCUDA(stream, 1);
Kernel<<<grid, block, 0, stream>>>(buffer->GetDevicePointer());
timeline->SignalCUDA(stream, 2);

ObjPtr<ICommandContext> consume = device->CreateCommandContext();
consume->CmdAcquireCUDABuffer(buffer.Get());
// Record graphics/compute/transfer commands using buffer here.
SubmitInfo after;
after.cuda_waits = {{timeline.Get(), 2}};
after.cuda_signals = {{timeline.Get(), 3}};
ObjPtr<ISubmissionToken> consumed = queue->Submit(consume.Get(), after);
```

Signals must strictly increase in **GPU execution order**, including signals from
different APIs. Independent streams should use separate timelines; one Graphics
submission can wait on all of them. Sharing a timeline across independent
producers requires explicit ordering between their signals. A higher value
satisfies waits for lower values; assigning values on the CPU alone does not
establish the required GPU order.

Each dependency list permits one point per timeline. In a submission that both
waits and signals the same timeline, the signal value must exceed the wait value.
Queue submissions reject null/foreign timelines, duplicate points and zero signal
values. Values and dependencies across different submissions remain the caller's
responsibility. Avoid dependency cycles.

The queue retains the timeline until its accepted submission completes. For CUDA
operations, the caller keeps it alive until all queued waits/signals have
completed, for example by synchronizing the involved streams at final teardown.
Destroying an imported semaphore with outstanding CUDA operations is invalid.
The timeline is separate from the RHI's internal resource-retirement timeline;
CUDA never signals internal submission completion fences.

RHIX exposes `CUDABuffer`, `CUDAInteropTimeline`, owned timeline points in
`rhix::SubmitInfo`, and the corresponding command/device methods.

Image/Array interop and CUDA IPC/VMM allocation export are outside this change.

## Validation

Build `test_rhi` explicitly and run `--gtest_filter='*CUDAInterop*'`. The tests
cover device matching, unsupported adapters, fixed pointers with a non-aligned
logical size, invalid dependencies, RHIX multiple waits/signals, and two CUDA
streams modifying shared buffers across four rounds without intermediate CPU
waits. Every downloaded element is checked against the expected kernel result.

References: [CUDA external resource API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXTRES__INTEROP.html),
[Vulkan queue ownership transfers](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-queue-transfers),
[D3D12 GPU queue waits](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-wait).
