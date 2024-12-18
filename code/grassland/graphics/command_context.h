#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {
class CommandContext {
 public:
  virtual ~CommandContext() = default;
  virtual void BindProgram(Program *program) = 0;
  virtual void BindColorTargets(const std::vector<Image *> &images) = 0;
  virtual void BindDepthTarget(Image *image) = 0;
  virtual void BindVertexBuffers(const std::vector<Buffer *> &buffers) = 0;
  virtual void BindIndexBuffer(Buffer *buffer) = 0;

  virtual void CmdSetViewport(const Viewport &viewport) = 0;
  virtual void CmdSetScissor(const Scissor &scissor) = 0;
  virtual void CmdDrawIndexed(uint32_t index_count,
                              uint32_t instance_count,
                              uint32_t first_index,
                              uint32_t vertex_offset,
                              uint32_t first_instance) = 0;
  virtual void CmdClearImage(Image *image, const ClearValue &color) = 0;
  virtual void CmdPresent(Window *window, Image *image) = 0;

 protected:
};
}  // namespace grassland::graphics
