# Main / Assist queue validation

Implementation tested: `eda07217`, based on main `6ddcd48e` (PR109).
Validation date: 2026-11-10. The earlier PR111 is closed; this is a separate implementation.

## Requirement audit

| Requirement | Implementation and evidence |
| --- | --- |
| One borrowed timeline point | `rhi_timeline.h`: readonly ITimeline and plain Event. Token has one Event; no IEvent, TimelinePoint, submission waits, or duplicate completion value. |
| Stable logical queues | Device owns fixed Main/Assist objects; D3D12 DIRECT queues, Vulkan queue indices in one graphics+compute family, Metal command queues. Native identity tests verify queue sharing against native capacity and require separate timelines. |
| Context queue binding | Only Queue creates contexts; Submit uses the bound Queue pointer. Raw API tests release caller queue references and still submit successfully. |
| Submit-only dependency consumption | Queue normalizes borrowed waits; each native backend encodes them during Submit. Submission stores only context and completion token. Retirement list storage is allocated before native submission. |
| Independent retirement and CPU waits | A real GPU gate blocks Assist while Main retires without another submit. WaitAny succeeds, WaitAll remains blocked until the gate opens; duplicate timeline values test min/max semantics. |
| Swapchain isolation | Recording rejects Assist clear, rendering, download and present of swapchain images. Main still presents afterward. Vulkan acquisition waits are submitted internally before exposing the image; only WSI retains binary semaphores. |
| Demonstrable dependency | `demo/rhi_multiqueue`: Assist updates input, Main computes and downloads; event/no-event modes use the same workload. |
| Minimal interface review | Removed Device context factory and queue forwarding, IsNative-style capability placeholders, duplicate native identity and physical-queue alias helpers, and obsolete per-context D3D12 Win32 events. |

## Builds and tests

All builds use Ninja and the existing Debug/Release directory convention. Local
builds include test_rhi, rhi_hello, rhix_hello and the multiqueue demo. Remote Release
builds include the same targets; remote Debug builds include test_rhi and the demo.

| Machine / configuration | Scope | Result |
| --- | --- | --- |
| Windows, RTX 5080, Debug | Full RHI suite, validation and window tests enabled | 237 passed, 1 existing DXR validation failure out of 238 |
| Windows, Debug, after final Vulkan family selection change | Commands, timelines and windows | 36 passed |
| Windows, Release | Commands, timelines, windows and raw API | 38 passed, 2 Debug-only checks skipped |
| Meshy-PC-3090, RTX 3090, Linux Release | Full RHI suite, validation, explicit X11 under Xvfb | 125 passed, 1 Debug-only check skipped |
| Meshy-PC-3090, Release, after final Vulkan family selection change | Timelines and windows | 14 passed, 1 Debug-only check skipped |
| Meshy-PC-3090, Debug, final source | Full RHI suite, validation, explicit X11 under Xvfb | 131 passed, no skips |

Local Debug full-suite results preceded the final queue-family selection cleanup;
that cleanup was then rebuilt and exercised by the stated local/remote targeted
runs and the full remote Debug suite. The final changes to the cross-queue test
name and a comment do not change its behavior.

Both Debug and Release multiqueue demo runs produced the following observations:

| Backend / machine | Event mode | No-event mode |
| --- | --- | --- |
| D3D12 / RTX 5080 | 0 incorrect frames / 32 | 32 incorrect frames / 32 |
| Vulkan / RTX 5080 | 0 incorrect frames / 32 | 32 incorrect frames / 32 |
| Vulkan / RTX 3090 | 0 incorrect frames / 32 | 32 incorrect frames / 32 |

Each run checked 2,097,152 integers. Event mode had zero incorrect values;
no-event mode had 2,097,152 incorrect values in these runs. No-event intentionally
violates resource synchronization requirements; D3D12 Debug also reports hazard
1047. These observations are not a performance claim or a portable guarantee that
missing synchronization always produces visible errors.

## Limitations and existing failures

- Windows Debug `CompactedLargeBlasSupportsInPlaceUpdate/0` still reports D3D12
  validation error 923 for the compacted BLAS destination address range. The same
  failure is recorded in the earlier PR109 withdrawal validation; this branch does
  not change the D3D12 acceleration structure implementation. It is not counted as
  passing or suppressed.
- Merely launching under Xvfb does not force this project's automatic window-system
  selection to X11. An initial Linux run selected the existing Wayland session and
  crashed in `xkb_state_key_get_layout` via `_glfwGetScancodeNameWayland`, during the
  synthetic ImGui key callback test. Explicit `LONGMARCH_WINDOW_SYSTEM=x11` passes.
  Wayland runtime validation is not claimed; no root-cause fix is included.
- Initial validation did not include macOS; see the macOS follow-up below.
- Collector native waits have a 50 ms bound for shutdown, failures and submission
  publication races. CPU retirement may therefore lag the GPU timeline; it does
  not require another Submit to make progress.
- Main and Assist share a native Vulkan queue when the selected graphics+compute
  family has only one queue. Timelines remain separate; native host operations
  share a mutex. Independent GPU progress is not guaranteed.

See [queue semantics](../rhi-queues.md) and the
[demo instructions](../../demo/rhi_multiqueue/README.md).

## Queue-owned submission follow-up

Submission records, dependency encoding, and retirement now belong to Queue.
Device retains only collector lifecycle control and multi-timeline waiting. Queue
uses its submission mutex for list handoff as well as native submit ordering;
completion polling and resource/token retirement run outside it. Unfinished records
return to their owning Queue without allocation.

The follow-up passed 58 local Windows Debug tests and 37 Meshy-PC-3090 Debug tests,
with validation and window tests enabled (explicit X11/Xvfb remotely). Coverage
includes Commands, Timeline, VulkanTimeline, Window, Raw API, and Download tests.
Both native builds passed; pre-commit checks passed. The platform limitations above
remain unchanged.


## macOS follow-up (2026-11-10)

Apple M5, native Metal and MoltenVK. Main/Assist now share one native Vulkan queue
when only one is available in the selected family. The shared native queue uses
one submission mutex; logical queues retain separate timelines and retirement
lists. Two CPU threads each submit 128 contexts and verify independent timeline
values and completed tokens. Existing cross-queue upload/copy, window ownership,
and context lifetime tests run without an absent-Assist fallback.

Both CLion Debug and Release configurations build `test_rhi`, `demo_rhi_hello`,
`demo_rhix_hello`, and `demo_rhi_multiqueue`. Full RHI suites run sequentially with
`LONGMARCH_RHI_WINDOW_TESTS=1`:

| Configuration | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Debug | 220 | 0 | 15 |
| Release | 211 | 0 | 16 |

No Vulkan VUID/validation errors occurred in the final runs. Removed four unused,
unreleased raw queue references from format tests. D24S8 tests query Vulkan native
format support before allocation; supported formats still require allocation and
all behavior assertions to pass. Only unsupported D24S8 cases are omitted.
The independent-GPU-progress test skips a shared native queue; both logical queues
are still required, and their identity, mutex sharing and separate timelines are
checked against native queue capacity. Other existing skips cover unsupported
features and Debug-only contracts, not unexpected failures.

Metal and Vulkan Event demos each report 0/32 incorrect frames and 0/2,097,152
incorrect integers in both configurations. Without Events, Metal reports 32/32
incorrect frames in Debug and 11/32 in Release; shared-queue Vulkan reports 0/32
in both. Unsynchronized results remain observations, not a correctness guarantee.

This follow-up does not rerun the historical Windows/Linux results above.
