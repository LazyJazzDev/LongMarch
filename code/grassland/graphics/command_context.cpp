#include "grassland/graphics/command_context.h"

#include "grassland/graphics/buffer.h"
#include "grassland/graphics/core.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/window.h"

namespace grassland::graphics {

void CommandContext::CmdBindResources(int slot, const std::vector<Buffer *> &buffers, BindPoint bind_point) {
  std::vector<BufferRange> buffer_ranges(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    buffer_ranges[i] = buffers[i]->Range();
  }
  CmdBindResources(slot, buffer_ranges, bind_point);
}

void CommandContext::PushPostExecutionCallback(std::function<void()> callback) {
  post_execution_callbacks_.push_back(std::move(callback));
}

const std::vector<std::function<void()>> &CommandContext::GetPostExecutionCallbacks() const {
  return post_execution_callbacks_;
}

void CommandContext::PybindClassRegistration(py::classh<CommandContext> &c) {
}

}  // namespace grassland::graphics
