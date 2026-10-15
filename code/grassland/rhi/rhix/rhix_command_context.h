#pragma once
#include <array>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
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
};

}  // namespace grassland::rhix
