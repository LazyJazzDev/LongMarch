# RHI native validation


`CreateCore(api, CoreDesc{enable_debug})` selects native D3D12/Vulkan validation.
Omitting the descriptor enables it when `NDEBUG` is undefined and disables it otherwise;
callers can explicitly pass `{false}` for profiling or `{true}` for Release diagnostics.
Requested but unavailable validation fails Core creation. D3D12 Cores alive at the same
time must use the same setting; the last Core disables the layer after caller-owned
Devices have been released. Vulkan validation is configured per instance. External
validation tools/environment overrides remain external; Metal validation still uses
native tools/environment settings.
