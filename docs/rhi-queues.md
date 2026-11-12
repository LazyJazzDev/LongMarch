# RHI queues, timelines, and completion

A Device provides stable Main and Assist queues through `GetQueue`. Both support
graphics, compute, and transfer commands; swapchain images are Main-only.
Create contexts through their queue. Each context permanently belongs to that
queue and is single-use; submit it with `context->Submit(info)`.

Vulkan prefers two native queues in one graphics+compute family. If only one is
available, Main and Assist share the native queue and its submission mutex but
keep separate timelines. D3D12 uses two DIRECT queues; Metal uses two command
queues. Logical queue identity does not guarantee independent GPU execution.

## Events and dependencies

`ITimeline` exposes completion queries and CPU waits. `Event { timeline, value }`
is a borrowed timeline point; its Device must outlive every use of the Event.

Pass producer completion Events in `SubmitInfo::waits` for cross-queue GPU
dependencies. Repeated waits on one timeline use the greatest value. CPU submit
order alone does not synchronize different queues. When logical queues share a
native queue, submit producers before consumers to avoid blocking a producer
behind its own consumer.

A completed Event means the GPU reached that point. A completed SubmissionToken
also means CPU-side retirement and result processing finished. Wait for the
download token before reading downloaded CPU storage; waiting for its Event alone
is insufficient. Submitted work retains its resources through retirement even
if the caller releases the context or token.

`Device::WaitEvents(events, wait_all)` supports All/Any waits. Duplicate timelines
use the maximum value for All and the minimum for Any. Empty All succeeds; empty
Any returns false. Timeline waits may precede the corresponding signal submission.

## Completion and presentation

Each Queue owns its pending submissions. A Device collector retires completed
work without requiring another Submit. Submission records are registered before
native submission; rejected work is removed. Completion processing runs outside
the queue submission lock, so blocked Assist work does not prevent completed Main
work from retiring. Shutdown wakes the collector and drains pending work.

Metal uses MTLEvent for GPU dependencies and a separate MTLSharedEvent for host
progress. Completion callbacks publish host progress or failure and wake the
collector, without polling or repeated collector listener registration.

Preparing the next swapchain image occurs outside the submission lock and does
not change the current submission result. Vulkan consumes binary acquisition
semaphores on Main before exposing the image; the swapchain manages WSI semaphore
reuse internally.

See `demo/rhi_multiqueue` for an upload/compute/readback example and an intentionally
unsynchronized comparison.
