# TUI message recorder

Build/run `demo_tui`. This standalone application derives from `snowberg::tui::Core`
and links only `snowberg_tui`; it does not initialize a graphics backend.

In CLion, choose the shared **TUI Messages** run configuration (terminal emulation
is enabled). Alternatively enable **Emulate terminal in the output console** on
the generated `demo_tui` configuration. A regular pipe/output console prints an
explanation and exits; the editor requires a terminal.

The upper pane shows session-local recorded messages. The lower pane has a
contrasting background, a `>` prompt and a blinking block cursor. It starts at
three rows (padding/label, editor, help), grows with multiline input, and scrolls
internally when it reaches its height limit. Resize reflows both panes. Very
small terminals show a resize hint until at least 12 columns and 6 rows are
available, without discarding the draft/history.

| Input | Action |
| --- | --- |
| Enter | Append the draft to history, follow its end, clear the input |
| Shift+Enter | Insert a newline (requires modified-key reporting) |
| Alt+Enter / Ctrl+J | Alternative newline keys for legacy terminals |
| Left / Right | Move across characters and line boundaries |
| Up / Down | Move between displayed input lines, preserving the preferred column |
| Home / End | Start/end of the displayed input line |
| Backspace / Delete | Delete before/at the cursor; joins lines at a newline |
| Left click in input | Place the cursor at that character cell |
| Wheel / PageUp / PageDown | Scroll message history |
| Esc / Ctrl+C | Exit and restore the terminal |

Input supports Unicode, wide CJK characters and common combining marks. Complex
emoji grapheme clusters retain the underlying TUI library's width limitations.
Bracketed paste inserts a whole block (including newlines) without sending it;
Enter explicitly sends it afterward. No network requests are made and messages
are not persisted automatically. `--dump` prints recorded messages after exit;
`--help` prints usage.

The demo requests CSI-u/Kitty key disambiguation, xterm modifyOtherKeys and
bracketed paste via `Core::EnableEnhancedInput()`. Hosts may ignore unsupported
requests. If Shift+Enter and Enter both arrive as the same CR byte, an application
cannot distinguish them; use an alternative newline key or configure the host to
send `ESC[13;2u` for Shift+Enter. Windows native input preserves Shift directly.
The terminal keyboard/paste modes are disabled/popped during teardown, before
input draining and ordinary-screen restoration.

On the tested macOS CLion Run console, Shift+Enter arrived as ordinary Enter
even after protocol requests. Alt+Enter was verified to insert a newline; use
that shortcut in this host. CSI-u and modifyOtherKeys Shift+Enter are covered by
protocol tests, not claimed as working in CLion's current emulator.
