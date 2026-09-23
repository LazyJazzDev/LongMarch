# Window magnification input

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
