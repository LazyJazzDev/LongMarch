#include "grassland/graphics/command_context.h"

#include <stdexcept>

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/buffer.h"
#include "grassland/graphics/core.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/program.h"
#include "grassland/graphics/sampler.h"
#include "grassland/graphics/window.h"

namespace grassland::graphics {

void CommandContext::CmdBindResources(int slot, const std::vector<Buffer *> &buffers, BindPoint bind_point) {
  std::vector<BufferRange> buffer_ranges(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    buffer_ranges[i] = buffers[i]->Range();
  }

  CmdBindResources(slot, buffer_ranges, bind_point);
}

void CommandContext::OnCompleted(std::function<void()> callback) {
  if (submission_started_)
    throw std::logic_error("cannot add completion callback after submission");
  post_execution_callbacks_.push_back(std::move(callback));
}

void CommandContext::BeginSubmission(Core *core) {
  if (GetCore() != core)
    throw std::invalid_argument("command context belongs to a different core");
  if (submission_started_)
    throw std::logic_error("command context already submitted");
  submission_started_ = true;
}

std::vector<std::function<void()>> CommandContext::TakeCompletionCallbacks() {
  return std::move(post_execution_callbacks_);
}

}  // namespace grassland::graphics
