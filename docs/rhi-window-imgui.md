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
auto draw_id = window->ImGuiEvent().RegisterCallback([] {
  ImGui::Begin("Controls");
  ImGui::TextUnformatted("Automatically drawn before Present");
  ImGui::End();
});
// Record scene commands, then end any active rendering scope.
if (!commands->CmdPresent(window))
  return false;
// Submit as usual. No manual ImGui NewFrame/Render or renderer calls.
```

The rhix equivalents use `window.EnableImGui()` and `window.ImGuiEvent()`.
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
also supports `--frames`, appears last in the selection menu, and skips headless
runs explicitly.

Each enabled window owns a separate ImGui context and default font atlas.
`CmdPresent` starts a frame, invokes ImGuiEvent, renders, then appends an overlay
pass with LOAD/STORE attachment operations before the native present transition.
It changes the command context's graphics bindings; callers recording more draws
must bind their own state again. Callbacks run at recording time, including for
recordings the caller subsequently abandons. Do not recursively present or
reconfigure that window from its UI callback. Disabling/re-enabling ImGui starts
with a fresh context. Registrations remain attached to Window until removed.

The caller's ImGui context is restored after initialization, input delivery and
rendering. `GetImGuiContext()` returns a borrowed pointer (null while disabled).
Input always reaches application callbacks, even if ImGui wants capture. For
application controls, select this context and inspect `WantCaptureMouse` or
`WantCaptureKeyboard`, restoring the previous context afterwards. ImGui's
multi-viewport feature and changes to the uploaded font atlas are not supported.

`ImTextureID` represents an `rhi::IImage*` from the same Device, cast through
`uintptr_t`; user images must remain valid until CmdPresent finishes recording.
They are sampled with a linear clamp sampler. The command context retains the
bound GPU resources, including fonts, after the CPU ImGui context is disabled.
Each recording snapshots vertices/indices into independent buffers; no hidden
per-frame GPU wait or reuse of a live upload occurs. EnableImGui waits once for
its initial font upload. Clip rectangles account for framebuffer scale, and the
renderer supports ImDrawCmd vertex/index offsets and reset-render-state callbacks.

The renderer uses only RHI operations and is shared by Metal, Vulkan and D3D12.
This branch starts from main's SDR Window API; HDR composition is outside this
change. Window, Device and Core retain their existing caller-owned lifetime rules.
