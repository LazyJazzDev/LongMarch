#pragma once
#include <map>

#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

class WebGPUCommandContext : public CommandContext {
 public:
  explicit WebGPUCommandContext(WebGPUCore *core);
  Core *GetCore() const override;

  void CmdBindProgram(Program *program) override;
  void CmdBindRayTracingProgram(RayTracingProgram *program) override;
  void CmdBindComputeProgram(ComputeProgram *program) override;

  void CmdBindVertexBuffers(uint32_t first_binding,
                            const std::vector<Buffer *> &buffers,
                            const std::vector<uint64_t> &offsets) override;
  void CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) override;
  void CmdBindResources(int slot,
                        const std::vector<BufferRange> &buffers,
                        BindPoint bind_point = BIND_POINT_GRAPHICS) override;
  using CommandContext::CmdBindResources;
  void CmdBindResources(int slot,
                        const std::vector<Image *> &images,
                        BindPoint bind_point = BIND_POINT_GRAPHICS) override;
  void CmdBindResources(int slot,
                        const std::vector<Sampler *> &samplers,
                        BindPoint bind_point = BIND_POINT_GRAPHICS) override;
  void CmdBindResources(int slot,
                        AccelerationStructure *acceleration_structure,
                        BindPoint bind_point = BIND_POINT_RAYTRACING) override;

  void CmdBeginRendering(const std::vector<Image *> &color_targets, Image *depth_target) override;
  void CmdEndRendering() override;

  void CmdSetViewport(const Viewport &viewport) override;
  void CmdSetScissor(const Scissor &scissor) override;
  void CmdSetPrimitiveTopology(PrimitiveTopology topology) override;
  void CmdDraw(uint32_t index_count, uint32_t instance_count, int32_t vertex_offset, uint32_t first_instance) override;
  void CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t vertex_offset,
                      uint32_t first_instance) override;
  void CmdClearImage(Image *image, const ClearValue &color) override;
  // Browser hosts present to their canvas themselves, from the backend's native texture.
  void CmdPresent(Window *window, Image *image) override;

  void CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth) override;
  void CmdDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) override;
  void CmdCopyBuffer(Buffer *dst_buffer,
                     Buffer *src_buffer,
                     uint64_t size,
                     uint64_t dst_offset = 0,
                     uint64_t src_offset = 0) override;

  // Ends open passes, runs remaining clears and returns the recorded commands.
  wgpu::CommandBuffer Finish();

  bool submitted = false;

 private:
  struct Resources {
    std::vector<BufferRange> buffers;
    std::vector<Image *> images;
    std::vector<Sampler *> samplers;
  };

  void EndPasses();
  void FlushClears();
  void PrepareDraw();
  void BindGroups(const std::vector<WebGPUBinding> &bindings,
                  const std::vector<wgpu::BindGroupLayout> &layouts,
                  BindPoint point);

  WebGPUCore *core_;
  wgpu::CommandEncoder encoder_;
  wgpu::RenderPassEncoder render_;
  wgpu::ComputePassEncoder compute_;
  Extent2D target_extent_{};
  std::map<Image *, ClearValue> pending_clears_;
  Viewport viewport_{};
  Scissor scissor_{};
  bool has_viewport_ = false, has_scissor_ = false;
  WebGPUProgram *program_ = nullptr;
  WebGPUComputeProgram *compute_program_ = nullptr;
  std::map<int, Resources> resources_[2];
  std::map<uint32_t, BufferRange> vertices_;
  BufferRange index_;
  PrimitiveTopology topology_ = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
};

}  // namespace grassland::graphics::backend
