# Terminal applications

Link `snowberg_tui` and include `snowberg/tui/tui.h` (or `tui_core.h`).
The library has no graphics dependencies. Applications derive from `Core`:

```cpp
class Example : public snowberg::tui::Core {
  void OnUpdate() override {
    SetForegroundColor(255, 220, 100);
    SetBackgroundColor(20, 20, 30);
    UpdateRow(0, 0, "Hello from TUI");
  }
  void OnKey(const snowberg::tui::KeyEvent &event) override {
    if (event.pressed && event.key == snowberg::tui::Key::Escape)
      Terminate();
  }
};

Example app;
const auto status = app.Run();
// RunResult::Unavailable: the caller may provide a plain-text interface.
// RunResult::Error: initialization, input, size query, or output failed.
```

## Responsibilities and lifecycle

- `tui_core.h/cpp`: platform terminal modes, lifecycle, event queue, callbacks,
  viewport queries and row/color output. Platform types are private.
- `tui_util.h/cpp`: key and mouse definitions, native/VT key mappings, capability
  checks, terminal size queries, incremental VT parsing, UTF-8 display-cell
  widths, safe slicing/prefix fitting and scrolling calculations. Complex emoji clusters and ambiguous-width characters
  are not fully normalized across terminals.
- `tui_menu.h/cpp`: the public `Menu` application derived from `Core`, with `Menu::Select` for the
  three hello demos, plus its plain-text fallback.

`Run()` checks the environment and enables terminal modes. It calls `OnStart`,
`OnResize(columns, rows)`, then `OnUpdate`. Each iteration reads a bounded input
batch, checks the viewport, dispatches queued `OnKey`/`OnMouse`/`OnPaste` events in order,
and calls `OnUpdate` once. Idle reads wait about 40 ms so applications can animate.
A resize callback precedes the batch's input callbacks. `Terminate()` stops further
input/update callbacks; `OnStop()` runs while output remains available, then
terminal modes are restored before `Run()` returns. Teardown first disables
mouse reporting, then discards queued/in-flight input while raw/no-echo input is
still active. It waits for 40 ms of quiet, capped at 250 ms under continuous
traffic, and performs a final platform input flush when restoring modes. Stale
reports are discarded, never sent to callbacks after termination. This bounded
drain cannot guarantee against a bridge delaying a report beyond the deadline. A later `Run()` can reuse the
Core; application state belongs to the subclass. Recursive Run is rejected.

All methods and callbacks use the Run thread. Only one application may own the
process terminal at a time; don't write unrelated console output concurrently.
Ordinary unavailable environments invoke no callbacks and emit no VT sequences.
Normal exits restore input modes, mouse reporting, cursor visibility, code page
(on Windows), and the primary screen. Forced process termination cannot restore
state. Callback exceptions are not translated into status codes; stack unwinding
restores terminal modes, but does not invoke `OnStop`.

## Input

`KeyEvent` preserves key, Unicode code point (for `Key::Character`), modifiers,
pressed/released state and repeat count. `MouseEvent` preserves zero-based
`column`/`row` cell coordinates, button, move/press/release/scroll action,
modifiers, and wheel steps (positive right/up). Core does not interpret these
as menu selection or confirmation. Applications implement their own hit testing.

Windows reads native console records and VT bridge input; POSIX reads terminal
bytes via poll/read. The VT decoder buffers fragmented UTF-8 and control
sequences and queues all complete events in arrival order. Standalone Escape
waits up to 80 ms to distinguish it from a sequence prefix. Traditional VT
keyboard input reports presses, not releases, and does not preserve all physical
key distinctions/modifiers. Do not rely on key-release support on POSIX.
Mouse tracking uses `1003` plus SGR `1006`, including hover/movement reports.
The queue is processed before rendering; movement events are not coalesced.

## Output

`SetForegroundColor(r,g,b)` and `SetBackgroundColor(r,g,b)` select RGB colors for
subsequent writes, using 24-bit VT color sequences. Terminal support determines
actual color fidelity. `ResetColors()` restores default color selection;
`SetInverse(bool)` controls reverse-video highlighting.

`UpdateRow(row, start_column, content)` uses **zero-based cell coordinates** and
re-queries the current viewport at write time. It overwrites only the prefix that
fits in the row, including the final column, without a newline or clearing the
remaining cells. Cursor repositioning cancels deferred right-margin wrapping.
`ClearRow(row)` explicitly erases a row using the selected background.

The return value is **consumed UTF-8 source bytes**, not cells or code points:
for `A中文B` with four cells remaining it writes `A中` and returns `4`.
A wide glyph that cannot fit is left unconsumed; no partial UTF-8 sequence is
written. Combining marks attached to an included base are retained; orphan
combining marks are consumed but omitted. Control characters and invalid bytes
are replaced by spaces so content cannot inject VT commands. Invalid coordinates
or an inactive Core return `0`; output/size-query failure returns `-1` and stops
Run with `Error`. Success means the output stream accepted the write, not a
terminal-side display acknowledgement. Each write is flushed immediately; Core
does not retain a framebuffer or automatically clear unchanged content.

## Selection menu and ordinary consoles

`Menu::Select(title, options, initial_selection)` returns a zero-based option index or
`std::nullopt` on cancellation, EOF, failure, or an empty menu. The menu implements
Up/Down, Enter, Esc/Ctrl+C/Ctrl+D, single-click selection, double-click confirmation
(within 400 ms), and wheel navigation through Core callbacks. Hover is delivered
by Core but does not change menu selection.

```cpp
snowberg::tui::Menu menu;
const auto choice = menu.Select("Example", {
    {"triangle", "Colored triangle"},
    {"texture", "Sampled texture"},
});
```

`Menu` is declared in `tui_menu.h` and included by `tui.h`. The same instance can
perform multiple selections; each call resets its options, result, selection,
scrolling and click state. There is no namespace-level `Select` function.

Each option occupies one row. Short viewports scroll vertically. Narrow rows
clip safely; a selected long label pauses one second, scrolls at 12.5 cells per
second, pauses at the end, and repeats. Only changed rows are rewritten. The menu
leaves the last column unused; the general Core API can write it.

If Core reports `Unavailable`, Menu::Select prints options sequentially and accepts an
option's one-based number or name. Invalid input retries; q/quit/Esc/EOF cancels.
The demos retain their direct CLI options such as `--module`.

Detection is capability-based: pipes, missing/empty/dumb TERM on POSIX, failed
size queries, or viewports smaller than two rows/columns use the text interface.
On POSIX an initially undersized PTY is retried for up to 500 ms because CLion
may initialize its terminal widget after starting the process. Ready terminals
start immediately. No modes change or VT output occurs during that wait.

For interactive CLion Run, enable **Emulate terminal in the output console**;
this differs from the Terminal tool window. A host advertising full capabilities
without implementing them cannot be reliably detected from handles/size alone.

## Validation

Build `test_tui`, `tui_menu_probe`, and `tui_core_probe`. `test_tui` covers text
width/clipping, byte counts, key mappings, queued/fragmented input, mouse
coordinates and malformed sequences. On POSIX run:

```sh
python3 test/tui/test_pty.py <tui_menu_probe> [hello-demo ...]
```

The Core probe must be beside the menu probe. The tests check colors, callback
order, termination with queued input, byte counts, edge coordinates, terminal
restoration, delayed viewport readiness, fallback, and menu interaction. On
macOS optional hello executables test one-frame Metal launches. Darwin's transient
PENDIN flag is excluded from configuration comparison. Protocol injection does
not prove a terminal frontend's physical mouse forwarding.

`test_console.py` and `test_conpty.py` provide native Windows and ConPTY menu
regressions. These require native Windows; the Core refactor has not yet been
run there. Linux also shares the POSIX code but was not tested in this change.

## Message editor demo

`demo/tui` provides `demo_tui`, a standalone multiline message recorder derived
from Core. The shared CLion **TUI Messages** configuration enables emulation.
See [the demo README](../../../demo/tui/README.md) for editing shortcuts and
Shift+Enter host compatibility. `EnableEnhancedInput()` opts into CSI-u / xterm
modified keys and bracketed paste; paste is delivered as one `OnPaste` event so
embedded newlines cannot accidentally submit messages. This is opt-in; Menu
does not enable the extra terminal modes.
