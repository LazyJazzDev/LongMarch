# RHI native validation


`CreateCore(api, CoreDesc{enable_debug})` selects native D3D12/Vulkan validation.
Omitting the descriptor enables it when `NDEBUG` is undefined and disables it otherwise;
callers can explicitly pass `{false}` for profiling or `{true}` for Release diagnostics.
Requested but unavailable validation fails Core creation. The caller must ensure only one Core exists at a time
and release its children before releasing it. D3D12 Core owns the debug controller
and disables the layer on destruction after caller-owned Devices have been released. Vulkan validation is configured per instance. External
validation tools/environment overrides remain external; Metal validation still uses
native tools/environment settings.

D3D12 Devices forward native warning/error messages (including message IDs) to the
same console logger used by Vulkan through ID3D12InfoQueue1 callbacks. Callbacks are
unregistered when the Device is destroyed. Older runtimes without this interface
retain native debugger output and emit a console warning about unavailable forwarding.
