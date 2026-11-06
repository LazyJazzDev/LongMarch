# Queue timelines and rendering dependencies

This demo uses a Transfer queue to update vertex colors and a Graphics queue to render and read back a 64 x 64 image. Each frame rotates the expected color through red, green, and blue. Every output pixel is checked.

```sh
cmake --build cmake-build-release --target demo_rhi_multiqueue
# Add .exe on Windows. Use --d3d12, --vulkan, or --metal.
cmake-build-release/demo/rhi_multiqueue/demo_rhi_multiqueue --vulkan --frames 32 --output synchronized.ppm
cmake-build-release/demo/rhi_multiqueue/demo_rhi_multiqueue --vulkan --frames 32 --without-event --output unsynchronized.ppm
```

`--copies N` adds N pairs of 32 MiB GPU copies ahead of each small vertex update on the Transfer queue. The default is 32. This creates a scheduling difference without a CPU sleep or a host wait between update and render submissions. Both submissions are joined before the next frame reuses resources.

The normal mode enables backend validation and waits for the update event on the Graphics queue. It exits unsuccessfully for any incorrect pixel. `--without-event` deliberately removes a required cross-queue dependency and disables validation: it is an invalid synchronization experiment whose output is not guaranteed. Its mismatch count is an observation, not a portable test expectation or a performance measurement. PPM output contains the final frame's actual GPU readback.

## Event and result semantics

```cpp
ObjPtr<ICommandContext> upload = transfer->CreateCommandContext();
// Record updates...
ObjPtr<ISubmissionToken> updated = upload->Submit();
Event ready = updated->GetEvent();

SubmitInfo info;
info.waits = {ready};
ObjPtr<ICommandContext> render = graphics->CreateCommandContext();
// Record rendering...
ObjPtr<ISubmissionToken> rendered = render->Submit(info);
rendered->WaitHost();
```

- `Event` is a trivially copyable `{ITimeline* timeline, uint64_t value}` value. `GetEvent()` returns it by value. Copies retain nothing; keep the Timeline and Device valid until all uses, including submitted waits, finish.
- `Event::WaitHost()` waits for GPU synchronization progress. `ISubmissionToken::WaitHost()` additionally waits for Collector retirement and result publication. A signaled event alone does not prove that a submission succeeded or that a download token is ready.
- `IDevice::WaitEvents(events, true)` waits for All; `false` waits for Any. The list must be nonempty and belong to one Device. Query individual events to identify completed points after Any returns. RHIX supplies the corresponding wrappers.
- Only `IQueue::CreateCommandContext()` creates contexts. Each context is permanently bound to that Queue and submits through `ICommandContext::Submit(info)`; no queue argument is needed. Device caches one persistent Queue object per role, so releasing caller Queue handles does not change the binding.
- `ITimeline` exposes `GetCompletedValue(uint64_t&)` and `WaitHost(value)`; it does not expose signaling. A failed query returns false. RHIX uses the same Event and SubmitInfo types, without ownership wrappers. Release queues and owned timeline references before their Device.

## Fixed queues and retirement

Graphics and Transfer roles are established at Device creation. Each Queue owns its native queue, submission mutex, serial allocator, and an independent Timeline object. Timeline encapsulates the native Fence / Timeline Semaphore / SharedEvent and does not reference a Queue. Their native timelines have independent monotonically increasing values. Cross-queue waits are encoded on the GPU; Submit does not wait on the CPU for its dependencies.

One Collector thread waits for Any of the fixed queues' current completed values plus one, even if the next Submit does not exist yet. It queries progress and retires all eligible records, so a blocked queue does not hold back another queue's completed records. Retirement records own only the context, completion token, and failure state. The token stores the single completion Event; there is no separate TimelinePoint or duplicate completion point. Dependency Events are passed directly to native submission and discarded when Submit returns. Command contexts retain only their Queue binding. The Queue allocates a serial before native submission, creating an immutable Event. Retirement records are allocated before native submission and published under the submission lock. The Collector releases its payload reference before publishing a Token result. Native GPU completion wakes it; ordinary Submit does not notify it. Shutdown submits and drains empty timeline signals to wake an otherwise idle Collector.

- D3D12 creates two distinct DIRECT queues and two fences. Transfer is a role, not a promise to use a dedicated DMA engine. Existing general command lists and COMMON resource boundaries remain compatible. Swapchain images must use their actual associated Queue. Image operations validate this while recording, before encoding native commands; a different Queue with the same Graphics role is also rejected. CPU batch waits use `ID3D12Device1::SetEventOnMultipleFenceCompletion`.
- Vulkan uses two queue indices in one Graphics/Compute family, avoiding cross-family ownership transfers. Each has its own Timeline Semaphore; waits use `vkWaitSemaphores`. If that family exposes only one queue, the roles alias and `SharesPhysicalQueue()` reports it. The demo rejects this configuration rather than claiming a multiqueue result.
- Metal creates two command queues and two SharedEvents. GPU dependencies use Event waits/signals; completion handlers notify one shared CPU wait service and propagate command-buffer errors. Metal runtime validation requires a Mac.

Vulkan swapchain images can only be used on their associated Graphics Queue. Acquisition submits a semaphore-only wait at ALL_COMMANDS on that queue before returning the image; subsequent contexts inherit this dependency. Each acquire slot records the bridge submission value for safe semaphore reuse. Context submission does not scan image states to discover acquire waits. This costs one additional native submission per successful acquisition.

All internal Vulkan queue dependencies use Timeline Semaphores. Binary Semaphores remain only at the swapchain acquire/present boundary because Vulkan WSI requires them; they are not exposed as queue timeline Events.

CUDA interoperability is intentionally outside this PR, which starts from main independently of the CUDA Buffer PR.

CPU batch waits coalesce Event values by actual ITimeline identity (maximum for All, minimum for Any), without a two-queue limit. Collector watches the fixed queue timelines and reads completion Events from Tokens. Tests also create three independent native timelines, copy Events without retaining them, and clear dependency storage immediately after Submit.
