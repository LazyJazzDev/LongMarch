#include "cao_di/graphics/command_context.h"

#include "cao_di/graphics/buffer.h"
#include "cao_di/graphics/core.h"
#include "cao_di/graphics/image.h"
#include "cao_di/graphics/window.h"

namespace CD::graphics {

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

void CommandContext::PyBind(pybind11::module &m) {
  pybind11::class_<CommandContext, std::shared_ptr<CommandContext>> command_context(m, "CommandContext");
  command_context.def("cmd_present", &CommandContext::CmdPresent);
  command_context.def(
      "cmd_clear_image",
      [](CommandContext *context, Image *image, const ColorClearValue &clear_color) {
        ClearValue clear_value{};
        clear_value.color = clear_color;
        context->CmdClearImage(image, clear_value);
      },
      pybind11::arg("image"), pybind11::arg("clear_color"));
  command_context.def(
      "cmd_clear_image",
      [](CommandContext *context, Image *image, const DepthClearValue &clear_depth) {
        ClearValue clear_value{};
        clear_value.depth = clear_depth;
        context->CmdClearImage(image, clear_value);
      },
      pybind11::arg("image"), pybind11::arg("clear_depth"));
  command_context.def("submit", [](CommandContext *context) { context->GetCore()->SubmitCommandContext(context); });
}

}  // namespace CD::graphics
