# RHI queues, timelines, and completion

A Device creates a fixed set of queues. `GetQueue(QUEUE_TYPE_MAIN)` and
`GetQueue(QUEUE_TYPE_ASSIST)` return references to stable objects, not new wrappers.
Each role has its own monotonically increasing timeline. Vulkan prefers two native
queues in one graphics+compute family; when the family has only one queue, both
roles share that native queue and its host submission mutex. Both roles support graphics, compute, and
transfer commands. Swapchain images belong exclusively to Main.

Create contexts through their queue. A context permanently belongs to that exact
queue; `context->Submit(info)` requires no queue argument. Commands validate
resource/queue compatibility during recording in Debug. Release callers must obey
the same preconditions. Contexts remain single-use.

`ITimeline` exposes completion queries and CPU waiting. `Event { timeline, value }`
is a borrowed timeline point, without a reference count or an owning queue reference.
Its Device must outlive all uses. There is no separate IEvent or TimelinePoint.

`SubmitInfo::waits` supplies GPU dependencies. Submit consumes the values and emits
native queue waits, deduplicating repeated timelines by their greatest value. The
retirement record does not retain the waits. The returned SubmissionToken owns one
completion Event and its result state, not a duplicate timeline point. Accepted GPU
work retains its resources through retirement even when the caller releases its
context or token. Retirement storage is allocated before native submission.

A completed Event means the GPU reached that point. SubmissionToken completion also
means CPU-side retirement and result processing finished; waiting for an Event is
not a substitute for waiting for a download token before reading its CPU storage.
`Device::WaitEvents(events, wait_all)` waits for GPU timeline progress, with All/Any
semantics. Duplicate timelines use the maximum value for All and minimum for Any.
An empty All succeeds; an empty Any returns false.

Each Queue owns its pending Submission list and performs its own submission and
retirement. Each Device starts one collector after initializing its native resources
and all queues. Each iteration queries each timeline once, reuses that completed
value to retire submissions, and waits for the next value. Queue locking protects native submit
order and list handoff; completion polling, resource release, and token publication
run outside that lock. A blocked Assist submission does not prevent an already
completed Main submission from retiring. Independent GPU progress is only possible
when the native queues can execute independently; shared queues can block behind
an earlier wait. Submit producers before consumers rather than relying on a later
submission to unblock a shared native queue. The collector can wait for the next value
before that submission exists. Native waits use a bounded 50 ms interval to observe
shutdown, failures, and publication races without adding fake queue signals. Thus
resource retirement can lag GPU completion by up to one wait interval when a record
is published after the collector has taken its progress snapshot. No subsequent
Submit is required to trigger retirement.

Vulkan queues share a graphics+compute family to avoid ownership transfers for
ordinary resources. D3D12 uses two DIRECT queues; Metal uses two command queues and
SharedEvent timelines. Logical queue identity does not promise that the
hardware executes their work simultaneously.

Vulkan WSI still requires binary acquire/present semaphores. After acquisition, the
Main queue immediately consumes the acquire semaphore in a semaphore-only submit
before exposing the image. Later Main work inherits that ordering. Ordinary context
submission neither scans image states for acquisition waits nor retains acquisition
semaphores. Only the swapchain manages WSI binary semaphore reuse.

Cross-queue resource hazards remain the caller's responsibility. Use the producer's
completion Event when submitting dependent work; do not infer synchronization from
CPU submit order. See `demo/rhi_multiqueue` for an Event/no-Event comparison.

CUDA interop, externally created timelines, and changes to the HeadlessWindow design
are outside this change.
