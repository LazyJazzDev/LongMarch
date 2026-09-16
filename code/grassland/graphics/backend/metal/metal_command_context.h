#pragma once
#include <map>

#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalCommandContext : public CommandContext {
 public:
  explicit MetalCommandContext(MetalCore *core);
  ~MetalCommandContext() override;
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
  void CmdPresent(Window *window, Image *image) override;

  void CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth) override;
  void CmdDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) override;
  void CmdCopyBuffer(Buffer *dst_buffer,
                     Buffer *src_buffer,
                     uint64_t size,
                     uint64_t dst_offset = 0,
                     uint64_t src_offset = 0) override;

  void EndEncoder();
  MTL::CommandBuffer *Handle() const {
    return command_.get();
  }
  bool submitted = false;

 private:
  struct Resources {
    std::vector<BufferRange> buffers;
    std::vector<Image *> images;
    std::vector<Sampler *> samplers;
  };
  void BindStage(MetalStage &stage, const std::vector<MetalBinding> &layout, BindPoint point, bool vertex = false);
  void PrepareDraw();
  MetalCore *core_;
  NS::SharedPtr<MTL::CommandBuffer> command_;
  NS::SharedPtr<MTL::ComputeCommandEncoder> compute_;
  NS::SharedPtr<MTL::RenderCommandEncoder> render_;
  NS::SharedPtr<MTL::RenderPassDescriptor> render_pass_;
  MTL::Viewport viewport_{};
  MTL::ScissorRect scissor_{};
  bool has_viewport_ = false, has_scissor_ = false, attachmentless_ = false;
  MetalComputeProgram *compute_program_ = nullptr;
  MetalProgram *program_ = nullptr;
  std::map<int, Resources> resources_[2];
  std::map<uint32_t, BufferRange> vertices_;
  BufferRange index_;
  MTL::PrimitiveType topology_ = MTL::PrimitiveTypeTriangle;
};

}  // namespace grassland::graphics::backend
