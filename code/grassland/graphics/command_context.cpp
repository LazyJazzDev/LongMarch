#include "grassland/graphics/command_context.h"

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

void CommandContext::PushPostExecutionCallback(std::function<void()> callback) {
  post_execution_callbacks_.push_back(std::move(callback));
}

const std::vector<std::function<void()>> &CommandContext::GetPostExecutionCallbacks() const {
  return post_execution_callbacks_;
}

void CommandContext::PybindClassRegistration(py::classh<CommandContext> &c) {
  c.def("cmd_bind_program", &CommandContext::CmdBindProgram, py::arg("program"), "Bind a graphics program",
        py::keep_alive<1, 2>{});
  c.def("cmd_bind_vertex_buffers", &CommandContext::CmdBindVertexBuffers, py::arg("first_binding"), py::arg("buffers"),
        py::arg("offsets") = std::vector<uint64_t>{}, "Bind vertex buffers", py::keep_alive<1, 2>{});
  c.def("cmd_bind_index_buffer", &CommandContext::CmdBindIndexBuffer, py::arg("buffer"), py::arg("offset") = 0,
        "Bind index buffer", py::keep_alive<1, 2>{});
  c.def("cmd_begin_rendering", &CommandContext::CmdBeginRendering, py::arg("color_targets"),
        py::arg("depth_target") = nullptr, "Begin rendering");
  c.def("cmd_end_rendering", &CommandContext::CmdEndRendering, "End rendering");
  c.def(
      "cmd_set_viewport",
      [](CommandContext *ctx, float x, float y, float width, float height, float min_depth, float max_depth) {
        ctx->CmdSetViewport(Viewport{x, y, width, height, min_depth, max_depth});
      },
      py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), py::arg("min_depth") = 0.0f,
      py::arg("max_depth") = 1.0f, "Set viewport");
  c.def(
      "cmd_set_scissor",
      [](CommandContext *ctx, int x, int y, uint32_t width, uint32_t height) {
        ctx->CmdSetScissor(Scissor{x, y, width, height});
      },
      py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), "Set scissor");
  c.def("cmd_set_primitive_topology", &CommandContext::CmdSetPrimitiveTopology, py::arg("topology"),
        "Set primitive topology");
  c.def("cmd_draw", &CommandContext::CmdDraw, py::arg("index_count"), py::arg("instance_count") = 1,
        py::arg("vertex_offset") = 0, py::arg("first_instance") = 0, "Draw indexed primitives");
  c.def("cmd_draw_indexed", &CommandContext::CmdDrawIndexed, py::arg("index_count"), py::arg("instance_count") = 1,
        py::arg("first_index") = 0, py::arg("vertex_offset") = 0, py::arg("first_instance") = 0,
        "Draw indexed primitives");
  c.def(
      "cmd_clear_image",
      [](CommandContext *ctx, Image *image, const std::vector<float> &color) {
        if (color.size() != 4) {
          throw std::runtime_error("Clear color must have 4 components");
        }
        ctx->CmdClearImage(image, ClearValue{color[0], color[1], color[2], color[3]});
      },
      py::arg("image"), py::arg("color"), "Clear an image");
  c.def(
      "cmd_clear_image",
      [](CommandContext *ctx, Image *image, float depth) { ctx->CmdClearImage(image, ClearValue{depth}); },
      py::arg("image"), py::arg("depth"), "Clear a depth image");
  c.def("cmd_present", &CommandContext::CmdPresent, py::arg("window"), py::arg("image"), "Present an image to a window",
        py::keep_alive<1, 2>{});
}

}  // namespace grassland::graphics
