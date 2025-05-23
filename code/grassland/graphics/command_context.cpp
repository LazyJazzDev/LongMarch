#include "grassland/graphics/command_context.h"

#include "grassland/graphics/core.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/window.h"

namespace grassland::graphics {

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

}  // namespace grassland::graphics
