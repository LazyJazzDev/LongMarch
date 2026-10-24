# Window callbacks and ImGui

`IWindow` and `rhix::Window` expose the same callback registries as the basic
input events in `graphics::Window`: `ResizeEvent`, `FramebufferResizeEvent`,
`CursorEnterEvent`, `FocusEvent`, `MouseMoveEvent`, `MouseButtonEvent`,
`ScrollEvent`, `KeyEvent`, `CharEvent` and `DropEvent`. RegisterCallback returns
an ID for UnregisterCallback; higher priorities execute first. Callbacks may
unregister themselves or other callbacks. New callbacks run on the next dispatch.
Registration and dispatch are single-threaded operations on the window thread.

Logical window sizes and mouse positions use GLFW content coordinates;
framebuffer sizes use physical pixels. The mouse-button callback receives
(button, action, modifiers, x, y), and key receives (key, scancode, action,
modifiers). Drop paths are borrowed for the duration of the callback. Native
framebuffer resize/restore handling remains automatic. The GLFW user pointer is
available to the application. Register through these interfaces instead of
replacing the GLFW callbacks installed by Window.

```cpp
// rhi::IWindow* window, created from a Device
if (!window->EnableImGui())
  return false;
// Each frame: poll input, then update UI before acquiring/recording scene work.
if (!window->UpdateImGui([&] {
      ImGui::Begin("Controls");
      ImGui::TextUnformatted("Drawn before Present");
      ImGui::End();
    }))
  return false;
// UI feedback is available now. Apply scene/swapchain changes, then record
// scene commands and end any active rendering scope.
if (!commands->CmdPresent(window))
  return false;
// Submit as usual. UpdateImGui handles ImGui NewFrame/Render internally.
```

The rhix equivalents use `window.EnableImGui()` and `window.UpdateImGui(callback)`.
Initialization accepts a font file and pixel size:
`window->EnableImGui(true, "fonts/MyFont.ttf", 24.0f)` (rhix uses `.`).
The defaults are `nullptr` and 13 pixels. A null path selects the built-in font
at the requested size; a file font uses the ChineseFull glyph range, matching
graphics D3D12/Vulkan (the font must contain the desired glyphs). Font data is
copied and the atlas is built/uploaded once during initialization. Repeated enable
calls preserve the existing context and font; disable before reinitializing to
change either option. An unreadable file or nonpositive/nonfinite size returns
false. Runtime font-atlas changes are not supported.
The independent `imgui` module of both hello demos demonstrates an interactive
overlay and the built-in ImGui demo. Press I to toggle UI recording, P to toggle
VSync, and Escape to close. I skips UI calls inside the callback without disabling
or recreating the context: empty frames produce no overlay and never replay old
draw data. The `window` module remains a plain clear/present example.
Select `imgui` from the menu or pass `--module imgui`; `--all` includes it.
Headless runs explicitly skip this platform-input example.

`rhi_hello --module imgui_multi_window` creates two independent native windows:
one with a counter, text input and background editor, and another with an animated
plot, speed/amplitude sliders and pause control. Each window owns its UI context
and state. I toggles only the focused window's UI; Escape or the native close
button closes only that window. The remaining window keeps running. This module
also supports `--frames`, appears as item 10 in the selection menu, and skips headless
runs explicitly.

Each enabled window owns a separate ImGui context and default font atlas.
`UpdateImGui(callback)` starts a frame, invokes the supplied callback and calls ImGui::Render to prepare
CPU draw data. Call it on the window thread after input polling and before scene
command recording so UI feedback affects the same frame. Apply swapchain changes
after the update and before acquiring images. Disabled ImGui updates are successful
no-ops and do not invoke the callback. The callback executes synchronously and is
not retained; captures only need to remain valid during the call. Keep persistent
widget state outside a per-frame lambda and capture it by reference. Passing `{}`
prepares an empty UI frame without invoking a callback. Do not recursively call
UpdateImGui, CmdPresent or EnableImGui from the UI
callback.

`CmdPresent` only consumes prepared draw data and appends GPU overlay commands
with LOAD/STORE attachment operations before the native present transition. It
never invokes the UI callback. Each update is consumed by at most one recording, even
if recording fails or is abandoned; update again for a subsequent overlay. Calling
UpdateImGui again before consumption replaces pending data with the latest frame.
An empty update, or a present without an update, draws no UI and never replays old
content. ImDrawCmd callbacks remain part of renderer command recording; they are
separate from the callback supplied to UpdateImGui.

Overlay recording changes graphics bindings; callers recording more draws must
bind their own state again. Disabling/re-enabling ImGui discards pending draw data
and starts with a fresh context. UI callbacks are passed per update, with no registration.

The caller's ImGui context is restored after initialization, input delivery and
UI updates and overlay recording. The public window API does not expose the context.
Input always reaches application callbacks, even if ImGui wants capture. For
application controls, inspect `ImGui::GetIO().WantCaptureMouse` or
`WantCaptureKeyboard` inside `UpdateImGui(callback)` and store the result in application state. ImGui's
multi-viewport feature and changes to the uploaded font atlas are not supported.

`ImTextureID` represents an `rhi::IImage*` from the same Device, cast through
`uintptr_t`; user images must remain valid until CmdPresent finishes recording.
They are sampled with a linear clamp sampler. The command context retains the
bound GPU resources, including fonts, after the CPU ImGui context is disabled.
Each window reuses vertex/index buffers, allocating 50% extra capacity above the required size when growing, without shrinking.
Uploads snapshot CPU data into staging buffers and execute before drawing on the graphics queue.
Command contexts retain replaced buffers until their GPU work completes; no hidden
per-frame GPU wait or reuse of a live upload occurs. EnableImGui waits once for
its initial font upload. Clip rectangles account for framebuffer scale, and the
renderer supports ImDrawCmd vertex/index offsets and reset-render-state callbacks.

The renderer uses only RHI operations and is shared by Metal, Vulkan and D3D12.
The integration includes main's HDR Window API, but color-managed HDR UI
composition is outside this change. Window, Device and Core retain their existing
caller-owned lifetime rules.

### Demo font options

Both hello demos accept `--font PATH` and `--font-size PIXELS` for ImGui modules,
including the RHI multi-window module. Options apply at startup; omit `--font`
to use the built-in font and omit `--font-size` to use 13 pixels.
For example: `demo_rhi_hello --backend d3d12 --module imgui --font "C:/Windows/Fonts/arial.ttf" --font-size 24 --frames 120`.
The size must be finite and positive. An unreadable font file fails initialization.
Changing these options requires restarting the demo; they do not rebuild the atlas live.

### UI colors on HDR outputs

ImGui renders directly into the swapchain in every output color space. The fragment shader
converts SDR/sRGB UI colors and sampled custom images to the target encoding: linear Rec.709,
reference-white-scaled scRGB, or Rec.2020/ST2084 PQ. Alpha remains unchanged. Solid regions
retain the same color and reference-white meaning as SDR; alpha blending takes place directly
in the target encoding, so translucent controls and text edges need not match SDR blending.

Reference white follows the window's current scale and existing platform conventions.
Custom images retain their SDR sampling interpretation; there is no automatic HDR image
color-space detection. HDR mode changes preserve the ImGui context and widget state.
There are no intermediate images, background conversion passes or swapchain sampling requirements.
Both hello demos' `hdr` modules include a gray-ramp and colored-text panel; H cycles output
color spaces, and bounded `--frames` runs cycle them automatically. Physical appearance still
depends on display calibration and system color management.

### N-body controls and HDR

Both hello demos' `nbody` modules follow `nbody_cs`: particles accumulate linear radiance
in a reusable RGBA16F image, dense highlights desaturate toward white, and a fullscreen
pass encodes the result to SDR or the selected HDR output. HDR brightness is bounded by
the window's reported headroom. ImGui then draws directly into the swapchain. The particle
accumulation image is part of the scene renderer, not an intermediate ImGui composition image.

The control panel uses `assets/fonts/simhei.ttf` at 20 pixels and shows backend, particle
count, frame duration and estimated interaction throughput. Reset, pause/resume, logarithmic
time step and galaxy-count controls match the reference; galaxy count applies on reset.
HDR toggles to the preferred supported mode (linear scRGB where available), while Output
selects an explicit supported color space. VSync is also configurable. UI updates finish
before scene command recording, so pause, reset, time-step and output changes affect that frame.

Shortcuts: H toggles HDR, P toggles VSync, Space pauses/resumes, R resets, Escape closes.
Text input capture suppresses simulation shortcuts; dragging outside ImGui rotates the view.
Headless mode retains deterministic initialization, three in-flight frames and 60 warmup
frames, with SDR encoding and no window or UI. Throughput is an estimate based on whole-frame
time, not measured GPU FLOP/s. The reference's speed-distribution plot is not ported because
RHI currently has no public buffer readback API. macOS/Linux runtime validation is separate.
