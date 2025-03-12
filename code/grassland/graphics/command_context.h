#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {
class CommandContext {
 public:
  virtual ~CommandContext() = default;

  virtual Core *GetCore() const = 0;

  virtual void CmdBindProgram(Program *program) = 0;
  virtual void CmdBindRayTracingProgram(RayTracingProgram *program) = 0;
  virtual void CmdBindVertexBuffers(uint32_t first_binding,
                                    const std::vector<Buffer *> &buffers,
                                    const std::vector<uint64_t> &offsets) = 0;
  virtual void CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) = 0;
  virtual void CmdBeginRendering(const std::vector<Image *> &color_targets, Image *depth_target) = 0;
  virtual void CmdBindResources(int slot,
                                const std::vector<Buffer *> &buffers,
                                BindPoint bind_point = BIND_POINT_GRAPHICS) = 0;
  virtual void CmdBindResources(int slot,
                                const std::vector<Image *> &images,
                                BindPoint bind_point = BIND_POINT_GRAPHICS) = 0;
  virtual void CmdBindResources(int slot,
                                const std::vector<Sampler *> &samplers,
                                BindPoint bind_point = BIND_POINT_GRAPHICS) = 0;
  virtual void CmdBindResources(int slot,
                                AccelerationStructure *acceleration_structure,
                                BindPoint bind_point = BIND_POINT_RAYTRACING) = 0;
  virtual void CmdEndRendering() = 0;

  virtual void CmdSetViewport(const Viewport &viewport) = 0;
  virtual void CmdSetScissor(const Scissor &scissor) = 0;
  virtual void CmdSetPrimitiveTopology(PrimitiveTopology topology) = 0;
  virtual void CmdDraw(uint32_t index_count,
                       uint32_t instance_count,
                       int32_t vertex_offset,
                       uint32_t first_instance) = 0;
  virtual void CmdDrawIndexed(uint32_t index_count,
                              uint32_t instance_count,
                              uint32_t first_index,
                              int32_t vertex_offset,
                              uint32_t first_instance) = 0;
  virtual void CmdClearImage(Image *image, const ClearValue &color) = 0;
  virtual void CmdPresent(Window *window, Image *image) = 0;

  virtual void CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth) = 0;

  static void PybindModuleRegistration(pybind11::module &m);

 protected:
};
}  // namespace grassland::graphics
