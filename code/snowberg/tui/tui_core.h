#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "snowberg/tui/tui_util.h"

namespace snowberg::tui {

enum class RunResult { Completed, Unavailable, Error };

// One Core owns the process terminal during Run(). Use on one thread only.
// Callbacks execute on that thread; do not write to the console concurrently.
class Core {
 public:
  Core();
  virtual ~Core();
  Core(const Core &) = delete;
  Core &operator=(const Core &) = delete;

  // Opens the terminal, dispatches queued events, updates, and restores it before
  // returning. Unavailable invokes no callbacks and emits no VT sequences.
  RunResult Run();
  void Terminate();
  // Opt in before Run: request disambiguated keys and bracketed paste.
  // Unsupported terminal protocols are ignored by the host.
  void EnableEnhancedInput(bool enabled = true);
  int Columns() const;
  int Rows() const;
  void SetForegroundColor(uint8_t r, uint8_t g, uint8_t b);
  void SetBackgroundColor(uint8_t r, uint8_t g, uint8_t b);
  void ResetColors();
  void SetInverse(bool enabled);
  // Zero-based cells. Overwrites only the fitting prefix, without wrapping or
  // clearing the rest of the row. Returns consumed UTF-8 source bytes, 0 for
  // invalid coordinates/inactive Core, -1 on output failure. Complete glyphs
  // only; controls/invalid UTF-8 are replaced by spaces. Colors apply per call.
  int UpdateRow(int row, int start_column, std::string content);
  void ClearRow(int row);

 protected:
  virtual void OnStart() {
  }

  virtual void OnResize(int columns, int rows) {
  }

  virtual void OnKey(const KeyEvent &event) {
  }

  virtual void OnMouse(const MouseEvent &event) {
  }

  virtual void OnPaste(const PasteEvent &event) {
  }

  virtual void OnUpdate() {
  }  // Runs after an event batch, or on a ~40 ms idle tick.

  virtual void OnStop() {
  }  // Terminal still active; restoration follows.

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace snowberg::tui
