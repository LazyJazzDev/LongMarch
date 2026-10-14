#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi {
using namespace backend;

CommandContext::CommandContext() = default;
CommandContext::CommandContext(CommandContext &&) noexcept = default;
CommandContext &CommandContext::operator=(CommandContext &&) noexcept = default;
RecordedCommands::RecordedCommands() = default;
RecordedCommands::RecordedCommands(RecordedCommands &&) noexcept = default;
RecordedCommands &RecordedCommands::operator=(RecordedCommands &&) noexcept = default;

bool CommandContext::ClearColor(const Image &image, const std::array<float, 4> &color) {
  return *this && Access::As<ICommandContext>(*this).Clear(Access::Retain<IImage>(image), color);
}

bool CommandContext::Present(const Image &image) {
  return *this && Access::As<ICommandContext>(*this).Present(Access::Retain<IImage>(image));
}

RecordedCommands CommandContext::Finish() {
  if (!*this)
    return {};
  auto commands = Access::Retain<ICommandContext>(*this);
  Release();
  return commands->Seal() ? Access::Share<RecordedCommands>(commands) : RecordedCommands{};
}

}  // namespace grassland::rhi
