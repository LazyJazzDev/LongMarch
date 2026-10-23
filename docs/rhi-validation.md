# RHI native validation

`CreateCore(api, CoreDesc{enable_debug})` selects native backend diagnostics.
Omitting the descriptor enables them when `NDEBUG` is undefined and disables them otherwise;
callers can explicitly pass `{false}` for profiling or `{true}` for Release diagnostics.
The caller must ensure only one Core exists at a time and release its children before releasing it.

D3D12 Core owns the debug controller and disables the layer on destruction after
caller-owned Devices have been released. Vulkan validation is configured per instance.
Requested but unavailable D3D12/Vulkan validation fails Core creation.

Metal uses `MTLCommandBufferErrorOptionEncoderExecutionStatus` for every RHI command
buffer when enabled, including queue-idle markers. Disabled Cores use no enhanced
error options. Command-buffer failures log the native error description and, when
enabled, its diagnostic user information. This setting is per Core and does not modify process environment variables.
Metal API Validation is separate: enable it through Xcode or `MTL_DEBUG_LAYER=1`
before launching the process. `enable_debug=false` does not disable that external layer.
External validation tools/environment overrides remain external on all backends.

D3D12 Devices forward native warning/error messages (including message IDs) to the
same console logger used by Vulkan through ID3D12InfoQueue1 callbacks. Callbacks are
unregistered when the Device is destroyed. Older runtimes without this interface
retain native debugger output and emit a console warning about unavailable forwarding.
