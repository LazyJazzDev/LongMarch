#include "grassland/graphics/backend/metal/metal_command_context.h"

#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_acceleration_structure.h"
#include "grassland/graphics/backend/metal/metal_buffer.h"
#include "grassland/graphics/backend/metal/metal_core.h"
#include "grassland/graphics/backend/metal/metal_image.h"
#include "grassland/graphics/backend/metal/metal_program.h"
#include "grassland/graphics/backend/metal/metal_sampler.h"
#include "grassland/graphics/backend/metal/metal_window.h"

namespace grassland::graphics::backend {

namespace {
void CheckPoint(BindPoint point) {
  if (point != BIND_POINT_COMPUTE && point != BIND_POINT_GRAPHICS)
    throw std::runtime_error("Metal native ray tracing programs are unavailable; use Sparkium compute fallback");
}
}  // namespace

MetalCommandContext::MetalCommandContext(MetalCore *core) : core_(core) {
  MetalPool pool;
  command_ = NS::RetainPtr(core_->Queue()->commandBuffer());
  MetalCheck(command_.get(), nullptr, "commandBuffer");
}

MetalCommandContext::~MetalCommandContext() {
  EndEncoder();
}

Core *MetalCommandContext::GetCore() const {
  return core_;
}

void MetalCommandContext::EndEncoder() {
  render_pass_.reset();
  if (compute_) {
    compute_->endEncoding();
    compute_.reset();
  }
  if (render_) {
    render_->endEncoding();
    render_.reset();
  }
}

void MetalCommandContext::CmdBindProgram(Program *program) {
  program_ = dynamic_cast<MetalProgram *>(program);
  if (!program_)
    throw std::invalid_argument("expected MetalProgram");
}

void MetalCommandContext::CmdBindComputeProgram(ComputeProgram *program) {
  compute_program_ = dynamic_cast<MetalComputeProgram *>(program);
  if (!compute_program_)
    throw std::invalid_argument("expected MetalComputeProgram");
}

void MetalCommandContext::CmdBindRayTracingProgram(RayTracingProgram *) {
  CheckPoint(BIND_POINT_RAYTRACING);
}

void MetalCommandContext::CmdBindResources(int slot, AccelerationStructure *structure, BindPoint point) {
  if (point != BIND_POINT_COMPUTE)
    throw std::invalid_argument("Metal acceleration structures currently support compute bindings only");
  auto native = dynamic_cast<MetalAccelerationStructure *>(structure);
  if (!native)
    throw std::invalid_argument("expected Metal acceleration structure");
  resources_[point][slot] = {{}, {}, {}, structure};
}

void MetalCommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t) {
  CheckPoint(BIND_POINT_RAYTRACING);
}

void MetalCommandContext::CmdBindResources(int slot, const std::vector<BufferRange> &buffers, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {buffers, {}, {}};
}

void MetalCommandContext::CmdBindResources(int slot, const std::vector<Image *> &images, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {{}, images, {}};
}

void MetalCommandContext::CmdBindResources(int slot, const std::vector<Sampler *> &samplers, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {{}, {}, samplers};
}

void MetalCommandContext::BindStage(MetalStage &stage,
                                    const std::vector<MetalBinding> &layout,
                                    BindPoint point,
                                    bool vertex) {
  MetalPool pool;
  for (auto &[buffer_slot, encoder] : stage.arguments) {
    auto argument =
        NS::TransferPtr(core_->Device()->newBuffer(encoder->encodedLength(), MTL::ResourceStorageModeShared));
    MetalCheck(argument.get(), nullptr, "argument buffer allocation");
    encoder->setArgumentBuffer(argument.get(), 0);
    for (auto [slot, argument_index] : stage.resource_indices) {
      if (!stage.packed && slot != buffer_slot)
        continue;
      auto it = resources_[point].find(slot);
      if (it == resources_[point].end())
        throw std::runtime_error("unbound Metal resource set " + std::to_string(slot));
      auto &resources = it->second;
      auto &binding = layout.at(slot);
      MTL::ResourceUsage usage = MTL::ResourceUsageRead;
      if (binding.type == RESOURCE_TYPE_WRITABLE_IMAGE || binding.type == RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER)
        usage |= MTL::ResourceUsageWrite;
      auto use = [&](MTL::Resource *resource) {
        if (point == BIND_POINT_COMPUTE)
          compute_->useResource(resource, usage);
        else
          render_->useResource(resource, usage, vertex ? MTL::RenderStageVertex : MTL::RenderStageFragment);
      };
      size_t count = resources.buffers.size() + resources.images.size() + resources.samplers.size() +
                     (resources.acceleration_structure ? 1 : 0);
      if (count != binding.count)
        throw std::runtime_error("Metal resource count mismatch at set " + std::to_string(slot));
      if (resources.acceleration_structure) {
        if (binding.type != RESOURCE_TYPE_ACCELERATION_STRUCTURE)
          throw std::invalid_argument("Metal AS bound to a non-AS resource slot");
        auto structure = dynamic_cast<MetalAccelerationStructure *>(resources.acceleration_structure);
        encoder->setAccelerationStructure(structure->Handle(), argument_index);
        use(structure->Handle());
        for (const auto &child : structure->Children())
          use(child.get());
        // Capture the exact AS generation and dependencies until this submission completes.
        PushPostExecutionCallback([storage = NS::RetainPtr(structure->Handle()), children = structure->Children()] {});
      } else if (binding.type == RESOURCE_TYPE_ACCELERATION_STRUCTURE) {
        throw std::invalid_argument("Metal AS slot requires an acceleration structure");
      }
      for (size_t i = 0; i < resources.buffers.size(); ++i) {
        auto &range = resources.buffers[i];
        auto buffer = dynamic_cast<MetalBuffer *>(range.buffer);
        if (!buffer || range.offset > buffer->Size() || range.size > buffer->Size() - range.offset)
          throw std::invalid_argument("invalid Metal buffer range");
        encoder->setBuffer(buffer->Handle(), range.offset, argument_index + i);
        use(buffer->Handle());
      }
      for (size_t i = 0; i < resources.images.size(); ++i) {
        auto image = dynamic_cast<MetalImage *>(resources.images[i]);
        if (!image)
          throw std::invalid_argument("expected MetalImage");
        encoder->setTexture(image->Handle(), argument_index + i);
        use(image->Handle());
      }
      for (size_t i = 0; i < resources.samplers.size(); ++i) {
        auto sampler = dynamic_cast<MetalSampler *>(resources.samplers[i]);
        if (!sampler)
          throw std::invalid_argument("expected MetalSampler");
        encoder->setSamplerState(sampler->state.get(), argument_index + i);
        // Samplers are indirect objects, not MTL::Resources. Retain through completion.
        PushPostExecutionCallback([state = sampler->state]() {});
      }
    }
    if (point == BIND_POINT_COMPUTE)
      compute_->setBuffer(argument.get(), 0, buffer_slot);
    else if (vertex)
      render_->setVertexBuffer(argument.get(), 0, buffer_slot);
    else
      render_->setFragmentBuffer(argument.get(), 0, buffer_slot);
  }
}

void MetalCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!compute_program_)
    throw std::runtime_error("no compute program bound");
  if (render_pass_)
    throw std::runtime_error("compute dispatch inside render pass");
  MetalPool pool;
  if (!compute_)
    compute_ = NS::RetainPtr(command_->computeCommandEncoder());
  compute_->setComputePipelineState(compute_program_->pipeline.get());
  BindStage(compute_program_->stage, compute_program_->bindings, BIND_POINT_COMPUTE);
  compute_->dispatchThreadgroups(MTL::Size(x, y, z), compute_program_->stage.threads);
  // BVH sorting/build/refit and traversal dispatches share storage buffers.
  compute_->memoryBarrier(MTL::BarrierScopeBuffers | MTL::BarrierScopeTextures);
}

void MetalCommandContext::CmdBeginRendering(const std::vector<Image *> &colors, Image *depth) {
  if (render_pass_)
    throw std::runtime_error("nested Metal render pass");
  EndEncoder();
  MetalPool pool;
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  for (size_t i = 0; i < colors.size(); ++i) {
    auto attachment = pass->colorAttachments()->object(i);
    attachment->setTexture(dynamic_cast<MetalImage *>(colors[i])->Handle());
    attachment->setLoadAction(MTL::LoadActionLoad);
    attachment->setStoreAction(MTL::StoreActionStore);
  }
  if (depth) {
    pass->depthAttachment()->setTexture(dynamic_cast<MetalImage *>(depth)->Handle());
    pass->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
    pass->depthAttachment()->setStoreAction(MTL::StoreActionStore);
  }
  render_pass_ = NS::RetainPtr(pass);
  attachmentless_ = colors.empty() && !depth;
}

void MetalCommandContext::CmdEndRendering() {
  EndEncoder();
}

void MetalCommandContext::CmdSetViewport(const Viewport &v) {
  if (!render_pass_)
    throw std::runtime_error("viewport outside render pass");
  viewport_ = MTL::Viewport{v.x, v.y, v.width, v.height, v.min_depth, v.max_depth};
  has_viewport_ = true;
  if (render_)
    render_->setViewport(viewport_);
}

void MetalCommandContext::CmdSetScissor(const Scissor &s) {
  if (!render_pass_ || s.offset.x < 0 || s.offset.y < 0)
    throw std::runtime_error("invalid Metal scissor");
  scissor_ = MTL::ScissorRect{NS::UInteger(s.offset.x), NS::UInteger(s.offset.y), s.extent.width, s.extent.height};
  has_scissor_ = true;
  if (render_)
    render_->setScissorRect(scissor_);
}

void MetalCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology topology) {
  static const MTL::PrimitiveType types[] = {MTL::PrimitiveTypeTriangle, MTL::PrimitiveTypeTriangleStrip,
                                             MTL::PrimitiveTypeLine, MTL::PrimitiveTypeLineStrip,
                                             MTL::PrimitiveTypePoint};
  if (topology < 0 || topology > PRIMITIVE_TOPOLOGY_POINT_LIST)
    throw std::invalid_argument("primitive topology");
  topology_ = types[topology];
}

void MetalCommandContext::CmdBindVertexBuffers(uint32_t first,
                                               const std::vector<Buffer *> &buffers,
                                               const std::vector<uint64_t> &offsets) {
  if (buffers.size() != offsets.size())
    throw std::invalid_argument("vertex buffer offsets");
  for (size_t i = 0; i < buffers.size(); ++i)
    vertices_.insert_or_assign(first + i, BufferRange(buffers[i], offsets[i]));
}

void MetalCommandContext::CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) {
  index_ = BufferRange(buffer, offset);
}

void MetalCommandContext::PrepareDraw() {
  if (!render_pass_ || !program_)
    throw std::runtime_error("draw needs a render pass and program");
  if (!render_) {
    MetalPool pool;
    // Delay encoder creation until the viewport is known. A pass that only
    // writes storage images still needs a rasterization extent on Metal.
    if (attachmentless_) {
      if (!has_viewport_ || viewport_.width <= 0 || viewport_.height <= 0)
        throw std::runtime_error("attachmentless Metal pass requires a viewport");
      render_pass_->setRenderTargetWidth(NS::UInteger(viewport_.originX + viewport_.width));
      render_pass_->setRenderTargetHeight(NS::UInteger(viewport_.originY + viewport_.height));
      render_pass_->setDefaultRasterSampleCount(1);
    }
    render_ = NS::RetainPtr(command_->renderCommandEncoder(render_pass_.get()));
    MetalCheck(render_.get(), nullptr, "renderCommandEncoder");
    // Grassland viewport/scissor state persists across passes in a context.
    if (has_viewport_)
      render_->setViewport(viewport_);
    if (has_scissor_)
      render_->setScissorRect(scissor_);
  }
  render_->setRenderPipelineState(program_->pipeline.get());
  render_->setDepthStencilState(program_->depth_state.get());
  render_->setCullMode(static_cast<MTL::CullMode>(program_->cull));
  render_->setFrontFacingWinding(MTL::WindingCounterClockwise);
  for (auto &[slot, range] : vertices_) {
    auto buffer = dynamic_cast<MetalBuffer *>(range.buffer);
    if (!buffer)
      throw std::invalid_argument("expected Metal vertex buffer");
    render_->setVertexBuffer(buffer->Handle(), range.offset, 16 + slot);
  }

  BindStage(program_->vertex_stage, program_->bindings, BIND_POINT_GRAPHICS, true);
  BindStage(program_->fragment_stage, program_->bindings, BIND_POINT_GRAPHICS, false);
}

void MetalCommandContext::CmdDraw(uint32_t count, uint32_t instances, int32_t first, uint32_t base_instance) {
  PrepareDraw();
  render_->drawPrimitives(topology_, NS::UInteger(first), count, instances, base_instance);
}

void MetalCommandContext::CmdDrawIndexed(uint32_t count,
                                         uint32_t instances,
                                         uint32_t first,
                                         int32_t base_vertex,
                                         uint32_t base_instance) {
  PrepareDraw();
  auto buffer = dynamic_cast<MetalBuffer *>(index_.buffer);
  if (!buffer)
    throw std::runtime_error("missing Metal index buffer");
  render_->drawIndexedPrimitives(topology_, count, MTL::IndexTypeUInt32, buffer->Handle(), index_.offset + first * 4,
                                 instances, base_vertex, base_instance);
}

void MetalCommandContext::CmdClearImage(Image *image, const ClearValue &value) {
  if (render_pass_)
    throw std::runtime_error("clear inside render pass");
  EndEncoder();
  MetalPool pool;
  auto metal = dynamic_cast<MetalImage *>(image);
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  if (IsDepthFormat(image->Format())) {
    pass->depthAttachment()->setTexture(metal->Handle());
    pass->depthAttachment()->setClearDepth(value.depth.depth);
    pass->depthAttachment()->setLoadAction(MTL::LoadActionClear);
    pass->depthAttachment()->setStoreAction(MTL::StoreActionStore);
  } else {
    auto attachment = pass->colorAttachments()->object(0);
    attachment->setTexture(metal->Handle());
    auto c = value.color;
    attachment->setClearColor(MTL::ClearColor(c.r, c.g, c.b, c.a));
    attachment->setLoadAction(MTL::LoadActionClear);
    attachment->setStoreAction(MTL::StoreActionStore);
  }
  command_->renderCommandEncoder(pass)->endEncoding();
}

void MetalCommandContext::CmdCopyBuffer(Buffer *dst,
                                        Buffer *src,
                                        uint64_t size,
                                        uint64_t dst_offset,
                                        uint64_t src_offset) {
  if (render_pass_)
    throw std::runtime_error("copy inside render pass");
  if (dst_offset > dst->Size() || size > dst->Size() - dst_offset || src_offset > src->Size() ||
      size > src->Size() - src_offset)
    throw std::out_of_range("Metal buffer copy");
  EndEncoder();
  MetalPool pool;
  auto blit = command_->blitCommandEncoder();
  blit->copyFromBuffer(dynamic_cast<MetalBuffer *>(src)->Handle(), src_offset,
                       dynamic_cast<MetalBuffer *>(dst)->Handle(), dst_offset, size);
  blit->endEncoding();
}

void MetalCommandContext::CmdPresent(Window *window, Image *image) {
  EndEncoder();
  dynamic_cast<MetalWindow *>(window)->Present(command_.get(), dynamic_cast<MetalImage *>(image));
}

}  // namespace grassland::graphics::backend
