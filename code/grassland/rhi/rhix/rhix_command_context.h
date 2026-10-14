#pragma once
#include <array>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class RecordedCommands : public ObjectContainer {
 public:
  RecordedCommands(rhi::IRecordedCommands *object) noexcept;
  rhi::IRecordedCommands *Get() const noexcept;
  RecordedCommands();
  RecordedCommands(RecordedCommands &&) noexcept;
  RecordedCommands &operator=(RecordedCommands &&) noexcept;
  RecordedCommands(const RecordedCommands &) = delete;
  RecordedCommands &operator=(const RecordedCommands &) = delete;
};

class CommandContext : public ObjectContainer {
 public:
  CommandContext(rhi::ICommandContext *object) noexcept;
  rhi::ICommandContext *Get() const noexcept;
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
};

}  // namespace grassland::rhix
