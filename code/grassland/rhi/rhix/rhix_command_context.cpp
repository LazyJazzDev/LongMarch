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

RecordedCommands::RecordedCommands() = default;
RecordedCommands::RecordedCommands(RecordedCommands &&) noexcept = default;
RecordedCommands &RecordedCommands::operator=(RecordedCommands &&) noexcept = default;

RecordedCommands::RecordedCommands(rhi::IRecordedCommands *object) noexcept : ObjectContainer(object) {
}

rhi::IRecordedCommands *RecordedCommands::Get() const noexcept {
  return static_cast<rhi::IRecordedCommands *>(GetObject());
}

bool CommandContext::ClearColor(const Image &image, const std::array<float, 4> &color) {
  return *this && Get()->ClearColor(image.Get(), color);
}

bool CommandContext::Present(const Image &image) {
  return *this && Get()->Present(image.Get());
}

RecordedCommands CommandContext::Finish() {
  if (!*this)
    return {};
  RecordedCommands commands = Get()->Finish();
  Release();
  return commands;
}
}  // namespace grassland::rhix
