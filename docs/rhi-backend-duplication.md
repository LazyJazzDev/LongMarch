# RHI backend workflow duplication audit

Audited revision: `631e9fcb0f41e1d54b6e081e28b5373a69da2910`.
Comparison main: `e233d265b4a9af8a6767ab2001b9357b38afbdeb`.

There are **48 confirmed complete workflow groups implemented three times**:
144 implementations, approximately 2,413 physical lines of function bodies
(including braces, comments and preprocessor branches). **45 groups previously
had a shared implementation in main; three were already duplicated before this
interface refactor.** This is a conservative workflow inventory, not a count of
every repeated getter, forwarding wrapper or similar source fragment.

The duplication is not required by the stateless-interface/final-implementation
design. Removing stateful implementation inheritance also removed shared
algorithms, although those are separate design decisions.

## Counting and verification

- Reviewed the Vulkan, D3D12 and Metal implementation directories, including
  `.cpp`, Metal `.mm`, and in-class definitions in headers/source files. Also
  checked the shared `backend_*` implementations for existing reuse.
- A group is one method/overload with a complete reusable workflow in all three
  backends. Four `BindResource` overloads count as four groups; a two-argument
  `RecordImGui` forwarding overload is excluded.
- Every listed group's full body matches after removing comments/whitespace
  and replacing the backend name (`Vulkan`, `D3D12`, `Metal`, including lowercase
  namespace spellings). This is stronger than merely sharing a method name.
  Types and hook calls still need appropriate parameters when extracting code.
- Pure getters, field-only initializers, one-call forwarding wrappers and
  constructors/destructors that only delegate are not included. Small but real
  workflows such as binding downloads, releasing a locked frame, or validating a
  download destination are included explicitly rather than hidden in a LOC rule.
- Checked the former main definition of every newly triplicated group. Binding
  version initialization previously lived in `BindingGroup::Version::Version`;
  window destruction lived in `Window::~Window`. The other groups retain their
  recognizable method names in the shared files.
- Platform-conditional code is counted as source duplication; this is not a
  statement that all three backends run on the same OS. No runtime changes or
  performance claim are made by this audit.

## Summary

| Area | Complete groups | Repeated responsibility |
| --- | ---: | --- |
| Device | 5 | Collector start/loop/shutdown, host-wait normalization, shader input validation |
| Queue | 2 | Submission transaction and completed-submission retirement |
| CommandContext | 9 | Rendering validation, sealing, submission wrapper, resource/download retention, ImGui/presentation helpers |
| BindingGroup + BindingVersion | 11 | Binding validation, copy-on-write, reference replacement and initialization |
| Graphics/compute Program | 2 | API-independent program/layout validation |
| Window | 15 | GLFW lifecycle, prepared-frame state, swapchain rebuild policy, ImGui and HDR policy |
| DownloadToken | 1 | Destination validation before backend readback copying |
| ScratchPool | 2 | Recent peak demand, capacity growth/shrink and pool clearing; pre-existing duplication |
| Core | 1 | Extracting public physical-device descriptions; pre-existing duplication |
| **Total** | **48** | **45 newly triplicated, 3 pre-existing** |

The highest-value extractions are Queue submission/retirement, Device collection,
binding updates, and Window lifecycle. For example, Queue::Submit duplicates the
entire dependency normalization, locking, serial assignment, pending-list
publication, native submission, rejection/exception cleanup and post-submit
presentation preparation. An ordering fix currently must be applied three times.

## Full inventory

The following links are pinned to the audited revision. Each row is one counted
group, not a family whose overloads are left implicit.

| # | Method / overload | Vulkan | D3D12 | Metal | Origin |
| ---: | --- | --- | --- | --- | --- |
| 1 | `Device::CreateShader` | [L341](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_device.cpp#L341) | [L307](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_device.cpp#L307) | [L134](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_device.cpp#L134) | Former shared implementation |
| 2 | `Device::StartCollector` | [L388](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_device.cpp#L388) | [L354](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_device.cpp#L354) | [L181](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_device.cpp#L181) | Former shared implementation |
| 3 | `Device::WaitEvents` | [L392](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_device.cpp#L392) | [L358](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_device.cpp#L358) | [L185](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_device.cpp#L185) | Former shared implementation |
| 4 | `Device::Collect` | [L413](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_device.cpp#L413) | [L379](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_device.cpp#L379) | [L206](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_device.cpp#L206) | Former shared implementation |
| 5 | `Device::ShutdownQueues` | [L436](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_device.cpp#L436) | [L402](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_device.cpp#L402) | [L229](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_device.cpp#L229) | Former shared implementation |
| 6 | `Queue::Submit` | [L59](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_queue.cpp#L59) | [L50](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_queue.cpp#L50) | [L67](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_queue.cpp#L67) | Former shared implementation |
| 7 | `Queue::CollectCompleted` | [L111](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_queue.cpp#L111) | [L102](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_queue.cpp#L102) | [L119](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_queue.cpp#L119) | Former shared implementation |
| 8 | `CommandContext::CmdDrawImGui` | [L982](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L982) | [L1040](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1040) | [L762](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L762) | Former shared implementation |
| 9 | `CommandContext::CmdPresentImage` | [L1008](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1008) | [L1066](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1066) | [L788](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L788) | Former shared implementation |
| 10 | `CommandContext::PrepareRendering` | [L1028](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1028) | [L1086](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1086) | [L818](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L818) | Former shared implementation |
| 11 | `CommandContext::RenderingMatches` | [L1067](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1067) | [L1125](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1125) | [L857](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L857) | Former shared implementation |
| 12 | `CommandContext::Submit` | [L1082](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1082) | [L1140](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1140) | [L872](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L872) | Former shared implementation |
| 13 | `CommandContext::BindDownloads` | [L1092](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1092) | [L1150](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1150) | [L882](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L882) | Former shared implementation |
| 14 | `CommandContext::RegisterDownload` | [L1101](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1101) | [L1159](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1159) | [L891](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L891) | Former shared implementation |
| 15 | `CommandContext::RetainResource` | [L1106](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1106) | [L1164](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1164) | [L896](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L896) | Former shared implementation |
| 16 | `CommandContext::Seal` | [L1120](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_command_context.cpp#L1120) | [L1178](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_command_context.cpp#L1178) | [L910](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_command_context.cpp#L910) | Former shared implementation |
| 17 | `BindingGroup::FindBinding` | [L415](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L415) | [L354](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L354) | [L340](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L340) | Former shared implementation |
| 18 | `BindingGroup::BindResource(BufferView)` | [L430](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L430) | [L369](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L369) | [L355](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L355) | Former shared implementation |
| 19 | `BindingGroup::BindResource(IImage*)` | [L465](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L465) | [L404](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L404) | [L390](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L390) | Former shared implementation |
| 20 | `BindingGroup::BindResource(ISampler*)` | [L490](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L490) | [L429](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L429) | [L415](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L415) | Former shared implementation |
| 21 | `BindingGroup::BindResource(ITopLevelAccelerationStructure*)` | [L513](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L513) | [L452](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L452) | [L438](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L438) | Former shared implementation |
| 22 | `BindingVersion::InitializeBindingVersion` | [L314](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L314) | [L255](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L255) | [L241](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L241) | Former shared implementation |
| 23 | `BindingVersion::IsComplete` | [L331](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L331) | [L272](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L272) | [L258](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L258) | Former shared implementation |
| 24 | `BindingVersion::Write(BufferView)` | [L339](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L339) | [L280](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L280) | [L266](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L266) | Former shared implementation |
| 25 | `BindingVersion::Write(IImage*)` | [L354](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L354) | [L295](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L295) | [L281](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L281) | Former shared implementation |
| 26 | `BindingVersion::Write(ISampler*)` | [L369](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L369) | [L308](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L308) | [L294](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L294) | Former shared implementation |
| 27 | `BindingVersion::Write(ITopLevelAccelerationStructure*)` | [L383](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_binding_group.cpp#L383) | [L322](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_binding_group.cpp#L322) | [L308](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_binding_group.cpp#L308) | Former shared implementation |
| 28 | `Program::ValidateLayout` | [L216](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_program.cpp#L216) | [L125](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_program.cpp#L125) | [L150](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_program.cpp#L150) | Former shared implementation |
| 29 | `ComputeProgram::ValidateLayout` | [L85](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_compute_program.cpp#L85) | [L60](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_compute_program.cpp#L60) | [L73](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_compute_program.cpp#L73) | Former shared implementation |
| 30 | `Window::EnableImGui` | [L435](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L435) | [L224](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L224) | [L27](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L27) | Former shared implementation |
| 31 | `Window::UpdateImGui` | [L452](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L452) | [L241](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L241) | [L44](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L44) | Former shared implementation |
| 32 | `Window::RecordPresentImage` | [L477](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L477) | [L266](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L266) | [L69](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L69) | Former shared implementation |
| 33 | `Window::RecordImGui(commands, image, color_space, white_scale)` | [L490](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L490) | [L279](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L279) | [L82](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L82) | Former shared implementation |
| 34 | `Window::GetNextImage` | [L514](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L514) | [L303](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L303) | [L106](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L106) | Former shared implementation |
| 35 | `Window::UpdateNextImage` | [L541](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L541) | [L330](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L330) | [L133](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L133) | Former shared implementation |
| 36 | `Window::ReleaseNextImage` | [L556](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L556) | [L345](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L345) | [L148](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L148) | Former shared implementation |
| 37 | `Window::RebuildSwapChain` | [L565](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L565) | [L354](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L354) | [L157](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L157) | Former shared implementation |
| 38 | `Window::SetPresentMode` | [L592](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L592) | [L381](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L381) | [L184](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L184) | Former shared implementation |
| 39 | `Window::SetColorSpace` | [L601](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L601) | [L390](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L390) | [L193](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L193) | Former shared implementation |
| 40 | `Window::GetHDRReferenceWhiteScale` | [L622](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L622) | [L411](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L411) | [L214](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L214) | Former shared implementation |
| 41 | `Window::GetHDRHeadroom` | [L632](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L632) | [L421](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L421) | [L224](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L224) | Former shared implementation |
| 42 | `Window::GetHDRReferenceWhiteNits` | [L664](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L664) | [L453](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L453) | [L256](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L256) | Former shared implementation |
| 43 | `Window::DestroyWindow` | [L709](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L709) | [L498](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L498) | [L301](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L301) | Former shared implementation |
| 44 | `Window::InitializeWindow` | [L715](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_window.cpp#L715) | [L504](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_window.cpp#L504) | [L307](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_window.cpp#L307) | Former shared implementation |
| 45 | `DownloadToken::CopyTo` | [L26](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_download_token_impl.h#L26) | [L26](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_download_token_impl.h#L26) | [L26](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_download_token_impl.h#L26) | Former shared implementation |
| 46 | `ScratchPool::Acquire` | [L9](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_scratch_pool.cpp#L9) | [L9](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_scratch_pool.cpp#L9) | [L9](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_scratch_pool.cpp#L9) | Pre-existing |
| 47 | `ScratchPool::Clear` | [L45](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_scratch_pool.cpp#L45) | [L44](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_scratch_pool.cpp#L44) | [L44](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_scratch_pool.cpp#L44) | Pre-existing |
| 48 | `Core::GetPhysicalDevices` | [L167](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/vulkan/vulkan_core.cpp#L167) | [L84](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/d3d12/d3d12_core.cpp#L84) | [L67](https://github.com/LazyJazzDev/LongMarch/blob/631e9fcb0f41e1d54b6e081e28b5373a69da2910/code/grassland/rhi/backend/metal/metal_core.cpp#L67) | Pre-existing |

## Partial similarities not included in the 48

The following are useful extraction candidates, but their complete bodies are
not interchangeable. These are examples of shared subflows, not another exact
duplicate count to add to 48.

| Candidate | Shared portion | Differences to preserve |
| --- | --- | --- |
| CmdPresent | Validate window/device/queue/thread, reject duplicate present, acquire/retain frame, record ImGui | Vulkan/D3D12 state transitions; Metal drawable; different present-list entries |
| UpdatePresentedImages | Iterate presented windows and prepare the next image | Vulkan first releases invalid swapchain images; messages/storage differ |
| CmdUploadBuffer / CmdCopyBuffer / CmdDownloadBuffer | Validate ranges/ownership, acquire or retain storage, delegate copying | Active render/compute encoder rules, native barriers/copy encoding, download completion handling |
| CmdBindGroup | Validate layout/device/completeness and retain the binding version | Vulkan descriptor sets, D3D12 descriptor tables, Metal argument buffers/residency |
| CmdBindProgram / CmdBindComputeProgram | Program ownership validation and context state setup | Pipeline/encoder setup and backend binding state |
| CmdBindVertexBuffer / CmdBindIndexBuffer | Buffer/range validation and lifetime retention | Native views, state transitions, binding encoding and per-draw preparation |
| BindingVersion::Clone | Allocate new version and preserve references/initialization flags | Vulkan copies initialized descriptor ranges, D3D12 copies descriptors, Metal copies argument buffers and rebinds its size buffer |
| DownloadToken::CopyData | Contiguous/row-pitched readback and stencil-plane handling | Vulkan/D3D12 copy stencil into packed bytes; Metal converts float depth to D24S8 |

`DownloadToken::CopyData` is already identical between Vulkan and D3D12, but is
not a three-way duplicate. Graphics ray-tracing `ValidateLayout` is also shared
in purpose between Vulkan and D3D12; Metal has no equivalent ray-tracing pipeline
implementation. These two-way cases are not used to inflate the table.

## Operations that should retain native implementations

- `WaitForProgress`, `WaitNativeEvents`, and `InterruptCollector`: Vulkan waits
  on timeline semaphores and a shutdown semaphore; D3D12 registers fence events
  and waits on Win32 handles; Metal uses completion/listener callbacks and a
  condition variable. Sharing `Collect` does not imply sharing these primitives.
- `SubmitNative` and `PollCompletion`: native submission/presentation differs.
  D3D12 may retry a retirement signal after failure; Metal inspects command-buffer
  status and publishes failure. They cannot be replaced with a generic completed
  counter comparison.
- `AcquireNativeFrame`, `RebuildNativeSwapChain`, resource creation, descriptor
  writes, acceleration-structure build/compaction and image-copy encoding have
  substantial native semantics. Common validation can be extracted separately.

## Existing reuse that should remain

`TemporalBufferPool` already shares its algorithm through a final class template
with backend types. Shader, SubmissionToken and ShaderBindingTable have common
final implementations. `ValidateBindingLayout`, `SameBindingLayout`, binding
mapping, GLFW helpers, Cocoa HDR handling, ImagePresenter and WindowImGui also
remain shared. They are not missing three implementations and are not counted.

## Extraction direction within the accepted design

Keep `Object<...>` and owned state in final concrete implementations. Share the
algorithm with internal free functions or templates; final member functions pass
only the containers, dependencies or operations that the algorithm needs. A
stateless interface's nonvirtual helper is another option when its existing
virtual capabilities are sufficient. Neither approach needs a stateful interface
base or another reference counter.

Do not introduce public mutable state accessors just to support extraction. The
final object can pass its private state to an internal helper while preserving
encapsulation. Native waiting, allocation and encoding remain narrow hooks.
For hot recording paths, prefer reusing already-known typed interfaces rather
than adding QueryInterface calls to access shared helpers.

Suggested order:

1. Extract Device collection and Queue submission/retirement while preserving
   publication order, lock boundaries, failure cleanup and object lifetimes.
2. Extract BindingGroup/BindingVersion transactions and program/rendering
   validation; preserve copy-on-write and retain-before-release behavior.
3. Extract Window/ImGui/HDR policies, keeping frame acquisition and native
   swapchain recreation in backend hooks.
4. Consolidate ScratchPool policy and readback helpers where measured complexity
   justifies it. Leave trivial accessors local.

This audit changes documentation only. It does not implement the extraction.
