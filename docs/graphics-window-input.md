# Window input and state

`grassland::graphics::Window` owns the native window callbacks. Application code
subscribes through its event managers instead of installing GLFW callbacks or
using a global application pointer. Each window routes events to its own
subscribers, including when ImGui chains the native callbacks.

| Interface | Coordinates / meaning |
| --- | --- |
| `GetSize()`, `GetWidth()`, `GetHeight()`, `ResizeEvent()` | Logical content size |
| `GetFramebufferSize()`, `FramebufferResizeEvent()` | Physical pixels; may be zero when minimized |
| `GetCursorPosition()`, `MouseMoveEvent()`, coordinates in `MouseButtonEvent()` | Logical content coordinates, top-left origin |
| `CursorEnterEvent()` | `true` on entry, `false` on exit |
| `FocusEvent()`, `IsFocused()` | Window keyboard focus; cancel active drags on focus loss |
| `IsKeyDown(key)`, `IsMouseButtonDown(button)` | Current input state, using the same GLFW codes as existing input events |
| `Focus()` | Request focus, subject to OS focus policy |
| `RequestClose()` | Set the close flag; leaves resources alive for orderly shutdown |
| `GetPosition()`, `SetPosition()`, `GetFrameSize()` | Logical desktop position and left/top/right/bottom decoration widths |
| `GetMonitorWorkArea()` | Work area `(x, y, width, height)` of the monitor with greatest overlap; primary monitor if none overlaps |

Call `Window::PollEvents()` on the main thread to process events for all windows.
Window operations and callbacks also belong on that thread. Use
`grassland::GetTimeSeconds()` for a monotonic clock independent of GLFW; it starts
at the first call. State queries require a live window. Keep subscription IDs
and unregister before destroying captured state.

Use mouse event coordinates directly when handling movement or button events.
To draw in physical pixels, scale logical positions by framebuffer size divided
by logical window size, guarding zero dimensions. Resize render targets with
`FramebufferResizeEvent()` or query the framebuffer size at a frame boundary;
ignore zero extents rather than allocating zero-sized images. Logical resize
events are not a substitute for pixel-size changes on high-DPI displays.

The new queries and events are also exposed through the Python `Window` binding
in snake_case. Vector queries return tuples. New event registration methods
return IDs accepted by their corresponding `unregister_*_event` methods.

## Demo audit

GOL and 2048 use per-window cursor/focus subscriptions and window state queries;
GOL cancels both grid and size-slider drags on focus loss. Their frame resources
are rebuilt at frame boundaries. N-body demos query mouse buttons through
`Window`; N-body, Draw & GUI, and Graphics Hello's resize module subscribe to
physical framebuffer resizing. Sparkium GUI uses window geometry/work-area
queries when fitting the film to the desktop. All demo event loops use
`Window::PollEvents()`; former GLFW clock calls use `GetTimeSeconds()`.

The only remaining direct GLFW calls under `demo/` are the six joystick/gamepad
device queries in `joystick_test`. These are global device diagnostics, not
window state or window callbacks, and are intentionally outside `Window`.

The desktop integration test is an explicit target (not part of headless tests):

```sh
cmake --build build --target test_window_input
build/test/graphics/test_window_input
```

It injects arguments through the installed native callback bridges to verify
multi-window routing, multiple subscriptions, unsubscription, focus transitions,
zero framebuffer extents, and surviving-window delivery after another window
closes. It also performs a real OS resize and checks logical/physical size
events and deferred close requests. Synthetic callback coverage does not replace
physical mouse, focus switching, or mixed-DPI desktop usability checks.

## Magnification

`grassland::graphics::Window` exposes native pinch input independently of the
rendering backend. Applications subscribe to `MagnifyEvent()` alongside
`ScrollEvent()` and remove their callback before their state is destroyed.

```cpp
auto id = window->MagnifyEvent().RegisterCallback(
    [](const grassland::graphics::MagnifyGesture &gesture) {
      // Apply gesture.scale around (gesture.x, gesture.y).
    });
// Before the subscriber is destroyed:
window->MagnifyEvent().UnregisterCallback(id);
```

`MagnifyGesture` contains:

- `scale`: positive incremental multiplier; 1.0 means no change. Multiply the
  current zoom by this value, rather than treating it as a cumulative scale.
- `x`, `y`: focus in logical window content coordinates, with a top-left origin,
  matching GLFW cursor coordinates. Convert to framebuffer pixels when needed.
- `phase`: `kBegin`, `kUpdate`, `kEnd`, or `kCancel`. Begin and end events may
  contain an increment. Cancel has scale 1.0 and does not undo earlier updates.

Callbacks run during normal window event processing on the window thread.
`SupportsMagnifyGestures()` reports whether a native adapter was installed;
this is not a guarantee that a touchpad is connected. It returns false after
window closure. The adapter is removed before the native window is destroyed.

## Platform support

| Window platform | Native magnification |
| --- | --- |
| macOS | Cocoa magnify events, for Metal and Vulkan alike |
| Windows / Linux | Not yet implemented; capability query returns false |

Keep Ctrl + scrolling as an application-level alternative. The window layer
never synthesizes a magnify event from scroll input: a driver may already map
pinch to Ctrl + scrolling, and producing both would apply zoom twice. Driver
emulation only works when GLFW observes both scrolling and the Ctrl key state.

The macOS integration test requires a logged-in desktop and can be run with:

```sh
cmake --build build --target test_window_gestures
build/test/graphics/test_window_gestures
```

It passes synthetic Cocoa events through the native adapter to check
multi-window routing, scale conversion, coordinates, phases, unsubscription,
and closed-window handling. This does not exercise OS gesture delivery or
replace a physical-touchpad usability check.
