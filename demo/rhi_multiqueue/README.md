# RHI multi-queue synchronization example

`demo_rhi_multiqueue <vulkan|d3d12|metal> <event|no-event>`

Assist uploads an input buffer. Main runs a compute shader (`output = input * 3 + 7`)
and downloads the result. Each of 32 iterations checks all 65,536 integers.
Main and Assist have separate timelines but may share one native queue. Event mode
works in either case; the example does not require independent GPU execution.

In `event` mode, Main submits with the upload token's completion Event in
`SubmitInfo::waits`. The CPU does not wait between the two submissions. Any incorrect
value makes the example fail.

In `no-event` mode, the same dependency is deliberately omitted. This is an invalid
cross-queue resource access experiment, not a supported way to render. Native
validation may report hazards. The result counts are observations, and zero incorrect
values would not establish correctness. A successful exit in this mode only means
the experiment completed, not that synchronization was valid.

Assist also copies an unrelated 16 MiB buffer 128 times before updating the input,
so Main is likely to read the previous iteration's input without the Event. This is
a scheduling aid, not a performance benchmark or a guarantee of failure. The CPU
waits for both submissions before reusing buffers in the next iteration.

Build with `cmake --build cmake-build-debug --target demo_rhi_multiqueue` (or the
existing Release configuration). Debug uses the default RHI validation settings.
No window or swapchain is involved; the example uses the normal buffer, command,
and compute APIs without introducing a headless window abstraction.
