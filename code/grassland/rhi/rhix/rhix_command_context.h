#pragma once
#include <array>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Window;

class CommandContext : public ObjectContainer {
 public:
  CommandContext(rhi::ICommandContext *object) noexcept;
  rhi::ICommandContext *Get() const noexcept;
  CommandContext();
  CommandContext(CommandContext &&) noexcept;
  CommandContext &operator=(CommandContext &&) noexcept;
  CommandContext(const CommandContext &) = delete;
  CommandContext &operator=(const CommandContext &) = delete;
  bool CmdClearColor(const Image &image, const std::array<float, 4> &color);
  bool CmdPresent(const Window &window);
};

}  // namespace grassland::rhix
