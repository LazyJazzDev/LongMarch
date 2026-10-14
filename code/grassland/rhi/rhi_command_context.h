#pragma once
#include <array>

#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi {
class RecordedCommands {
 public:
  RecordedCommands();
  RecordedCommands(RecordedCommands &&) noexcept;
  RecordedCommands &operator=(RecordedCommands &&) noexcept;
  RecordedCommands(const RecordedCommands &) = delete;
  RecordedCommands &operator=(const RecordedCommands &) = delete;
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::ICommandContext;
  ObjectRef object_;
  friend struct backend::Access;
};

class CommandContext {
 public:
  CommandContext();
  CommandContext(CommandContext &&) noexcept;
  CommandContext &operator=(CommandContext &&) noexcept;
  CommandContext(const CommandContext &) = delete;
  CommandContext &operator=(const CommandContext &) = delete;
  bool ClearColor(const Image &image, const std::array<float, 4> &color);
  bool Present(const Image &image);
  // Consumes this context, including on failure. This first presentation path
  // records one clear + present pair, or an empty queue submission.
  RecordedCommands Finish();
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::ICommandContext;
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
