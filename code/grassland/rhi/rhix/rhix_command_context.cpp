#include "grassland/rhi/rhix/rhix_command_context.h"

namespace grassland::rhix {
CommandContext::CommandContext() = default;
CommandContext::CommandContext(CommandContext &&) noexcept = default;
CommandContext &CommandContext::operator=(CommandContext &&) noexcept = default;

CommandContext::CommandContext(rhi::ICommandContext *object) noexcept : ObjectContainer(object) {
}

rhi::ICommandContext *CommandContext::Get() const noexcept {
  return static_cast<rhi::ICommandContext *>(GetObject());
}

bool CommandContext::CmdClearColor(const Image &image, const std::array<float, 4> &color) {
  return *this && Get()->CmdClearColor(image.Get(), color);
}

bool CommandContext::CmdPresent(const Image &image) {
  return *this && Get()->CmdPresent(image.Get());
}

}  // namespace grassland::rhix
