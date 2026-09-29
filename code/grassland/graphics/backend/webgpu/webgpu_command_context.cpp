#include "grassland/graphics/backend/webgpu/webgpu_command_context.h"

#include <algorithm>
#include <stdexcept>

#include "grassland/graphics/backend/webgpu/webgpu_buffer.h"
#include "grassland/graphics/backend/webgpu/webgpu_core.h"
#include "grassland/graphics/backend/webgpu/webgpu_image.h"
#include "grassland/graphics/backend/webgpu/webgpu_program.h"
#include "grassland/graphics/backend/webgpu/webgpu_sampler.h"

namespace grassland::graphics::backend {

namespace {
void CheckPoint(BindPoint point) {
  if (point != BIND_POINT_COMPUTE && point != BIND_POINT_GRAPHICS)
    throw std::runtime_error("WebGPU has no ray tracing");
}

WebGPUImage *Native(Image *image) {
  auto native = dynamic_cast<WebGPUImage *>(image);
  if (!native)
    throw std::invalid_argument("expected WebGPUImage");
  return native;
}

WebGPUBuffer *Native(Buffer *buffer) {
  auto native = dynamic_cast<WebGPUBuffer *>(buffer);
  if (!native)
    throw std::invalid_argument("expected WebGPUBuffer");
  return native;
}

// A range that reaches the end of the buffer covers the padded allocation: WGSL
// rounds uniform structs up to 16 bytes, beyond the engine's C++ struct sizes.
uint64_t BindingSize(const BufferRange &range) {
  auto buffer = Native(range.buffer);
  if (range.offset > buffer->Size())
    throw std::out_of_range("WebGPU buffer binding");
  if (range.size >= buffer->Size() - range.offset)
    return buffer->AllocatedSize() - range.offset;
  return range.size;
}
}  // namespace

WebGPUCommandContext::WebGPUCommandContext(WebGPUCore *core) : core_(core) {
  encoder_ = core_->Device().CreateCommandEncoder();
}

Core *WebGPUCommandContext::GetCore() const {
  return core_;
}

void WebGPUCommandContext::EndPasses() {
  if (render_) {
    render_.End();
    render_ = nullptr;
  }
  if (compute_) {
    compute_.End();
    compute_ = nullptr;
  }
}

void WebGPUCommandContext::FlushClears() {
  for (const auto &[image, clear] : pending_clears_) {
    auto native = Native(image);
    wgpu::RenderPassDescriptor pass{};
    wgpu::RenderPassColorAttachment color{};
    wgpu::RenderPassDepthStencilAttachment depth{};
    if (IsDepthFormat(image->Format())) {
      depth.view = native->View();
      depth.depthLoadOp = wgpu::LoadOp::Clear;
      depth.depthStoreOp = wgpu::StoreOp::Store;
      depth.depthClearValue = clear.depth.depth;
      pass.depthStencilAttachment = &depth;
    } else {
      color.view = native->View();
      color.loadOp = wgpu::LoadOp::Clear;
      color.storeOp = wgpu::StoreOp::Store;
      color.clearValue = {clear.color.r, clear.color.g, clear.color.b, clear.color.a};
      pass.colorAttachmentCount = 1;
      pass.colorAttachments = &color;
    }
    encoder_.BeginRenderPass(&pass).End();
  }
  pending_clears_.clear();
}

void WebGPUCommandContext::CmdBindProgram(Program *program) {
  program_ = dynamic_cast<WebGPUProgram *>(program);
  if (!program_)
    throw std::invalid_argument("expected WebGPUProgram");
}

void WebGPUCommandContext::CmdBindComputeProgram(ComputeProgram *program) {
  compute_program_ = dynamic_cast<WebGPUComputeProgram *>(program);
  if (!compute_program_)
    throw std::invalid_argument("expected WebGPUComputeProgram");
}

void WebGPUCommandContext::CmdBindRayTracingProgram(RayTracingProgram *) {
  CheckPoint(BIND_POINT_RAYTRACING);
}

void WebGPUCommandContext::CmdBindResources(int slot, const std::vector<BufferRange> &buffers, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {buffers, {}, {}};
}

void WebGPUCommandContext::CmdBindResources(int slot, const std::vector<Image *> &images, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {{}, images, {}};
}

void WebGPUCommandContext::CmdBindResources(int slot, const std::vector<Sampler *> &samplers, BindPoint point) {
  CheckPoint(point);
  resources_[point][slot] = {{}, {}, samplers};
}

void WebGPUCommandContext::CmdBindResources(int, AccelerationStructure *, BindPoint) {
  throw std::runtime_error("WebGPU has no acceleration structures");
}

void WebGPUCommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t) {
  CheckPoint(BIND_POINT_RAYTRACING);
}

void WebGPUCommandContext::BindGroups(const std::vector<WebGPUBinding> &bindings,
                                      const std::vector<wgpu::BindGroupLayout> &layouts,
                                      BindPoint point) {
  for (uint32_t slot = 0; slot < bindings.size(); ++slot) {
    auto found = resources_[point].find(int(slot));
    if (found == resources_[point].end())
      throw std::runtime_error("unbound WebGPU resource set " + std::to_string(slot));
    const auto &resources = found->second;
    wgpu::BindGroupEntry entry{};
    entry.binding = 0;
    if (!resources.buffers.empty()) {
      const auto &range = resources.buffers[0];
      entry.buffer = Native(range.buffer)->Handle();
      entry.offset = range.offset;
      entry.size = BindingSize(range);
    } else if (!resources.images.empty()) {
      entry.textureView = Native(resources.images[0])->View();
    } else if (!resources.samplers.empty()) {
      auto sampler = dynamic_cast<WebGPUSampler *>(resources.samplers[0]);
      if (!sampler)
        throw std::invalid_argument("expected WebGPUSampler");
      entry.sampler = sampler->Handle();
    } else {
      throw std::runtime_error("empty WebGPU resource set " + std::to_string(slot));
    }
    wgpu::BindGroupDescriptor descriptor{};
    descriptor.layout = layouts[slot];
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    auto group = core_->Device().CreateBindGroup(&descriptor);
    if (point == BIND_POINT_COMPUTE)
      compute_.SetBindGroup(slot, group);
    else
      render_.SetBindGroup(slot, group);
  }
}

void WebGPUCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!compute_program_)
    throw std::runtime_error("no compute program bound");
  if (render_)
    throw std::runtime_error("compute dispatch inside render pass");
  FlushClears();
  // Consecutive dispatches share one pass; WebGPU orders their storage accesses.
  if (!compute_)
    compute_ = encoder_.BeginComputePass();
  compute_.SetPipeline(compute_program_->Pipeline());
  BindGroups(compute_program_->Bindings(), compute_program_->Layouts(), BIND_POINT_COMPUTE);
  compute_.DispatchWorkgroups(x, y, z);
}

void WebGPUCommandContext::CmdBeginRendering(const std::vector<Image *> &colors, Image *depth) {
  if (render_)
    throw std::runtime_error("nested WebGPU render pass");
  EndPasses();
  std::vector<wgpu::RenderPassColorAttachment> attachments(colors.size());
  target_extent_ = {};
  for (size_t i = 0; i < colors.size(); ++i) {
    auto &attachment = attachments[i];
    attachment.view = Native(colors[i])->View();
    attachment.storeOp = wgpu::StoreOp::Store;
    auto clear = pending_clears_.find(colors[i]);
    attachment.loadOp = clear == pending_clears_.end() ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear;
    if (clear != pending_clears_.end()) {
      const auto c = clear->second.color;
      attachment.clearValue = {c.r, c.g, c.b, c.a};
      pending_clears_.erase(clear);
    }
    target_extent_ = colors[i]->Extent();
  }
  wgpu::RenderPassDepthStencilAttachment depth_attachment{};
  if (depth) {
    depth_attachment.view = Native(depth)->View();
    depth_attachment.depthStoreOp = wgpu::StoreOp::Store;
    auto clear = pending_clears_.find(depth);
    depth_attachment.depthLoadOp = clear == pending_clears_.end() ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear;
    if (clear != pending_clears_.end()) {
      depth_attachment.depthClearValue = clear->second.depth.depth;
      pending_clears_.erase(clear);
    }
    target_extent_ = depth->Extent();
  }
  // Other cleared images may be sampled by this pass. Clear them first.
  FlushClears();
  wgpu::RenderPassDescriptor pass{};
  pass.colorAttachmentCount = attachments.size();
  pass.colorAttachments = attachments.data();
  pass.depthStencilAttachment = depth ? &depth_attachment : nullptr;
  render_ = encoder_.BeginRenderPass(&pass);
  // Grassland viewport/scissor state persists across passes in a context.
  if (has_viewport_)
    CmdSetViewport(viewport_);
  if (has_scissor_)
    CmdSetScissor(scissor_);
}

void WebGPUCommandContext::CmdEndRendering() {
  EndPasses();
}

void WebGPUCommandContext::CmdSetViewport(const Viewport &v) {
  viewport_ = v;
  has_viewport_ = true;
  if (render_)
    render_.SetViewport(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
}

void WebGPUCommandContext::CmdSetScissor(const Scissor &s) {
  if (s.offset.x < 0 || s.offset.y < 0)
    throw std::runtime_error("invalid WebGPU scissor");
  scissor_ = s;
  has_scissor_ = true;
  if (!render_)
    return;
  // WebGPU rejects scissors outside the attachments; the visible area is the same.
  const uint32_t x = std::min<uint32_t>(s.offset.x, target_extent_.width),
                 y = std::min<uint32_t>(s.offset.y, target_extent_.height);
  render_.SetScissorRect(x, y, std::min(s.extent.width, target_extent_.width - x),
                         std::min(s.extent.height, target_extent_.height - y));
}

void WebGPUCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology topology) {
  if (topology < 0 || topology > PRIMITIVE_TOPOLOGY_POINT_LIST)
    throw std::invalid_argument("primitive topology");
  topology_ = topology;
}

void WebGPUCommandContext::CmdBindVertexBuffers(uint32_t first,
                                                const std::vector<Buffer *> &buffers,
                                                const std::vector<uint64_t> &offsets) {
  if (buffers.size() != offsets.size())
    throw std::invalid_argument("vertex buffer offsets");
  for (size_t i = 0; i < buffers.size(); ++i)
    vertices_.insert_or_assign(first + i, BufferRange(buffers[i], offsets[i]));
}

void WebGPUCommandContext::CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) {
  index_ = BufferRange(buffer, offset);
}

void WebGPUCommandContext::PrepareDraw() {
  if (!render_ || !program_)
    throw std::runtime_error("draw needs a render pass and program");
  render_.SetPipeline(program_->Pipeline(topology_));
  for (auto &[slot, range] : vertices_) {
    auto buffer = Native(range.buffer);
    render_.SetVertexBuffer(slot, buffer->Handle(), range.offset, buffer->AllocatedSize() - range.offset);
  }
  BindGroups(program_->Bindings(), program_->Layouts(), BIND_POINT_GRAPHICS);
}

void WebGPUCommandContext::CmdDraw(uint32_t count, uint32_t instances, int32_t first, uint32_t base_instance) {
  PrepareDraw();
  render_.Draw(count, instances, uint32_t(first), base_instance);
}

void WebGPUCommandContext::CmdDrawIndexed(uint32_t count,
                                          uint32_t instances,
                                          uint32_t first,
                                          int32_t base_vertex,
                                          uint32_t base_instance) {
  PrepareDraw();
  if (!index_.buffer)
    throw std::runtime_error("missing WebGPU index buffer");
  auto buffer = Native(index_.buffer);
  render_.SetIndexBuffer(buffer->Handle(), wgpu::IndexFormat::Uint32, index_.offset,
                         buffer->AllocatedSize() - index_.offset);
  render_.DrawIndexed(count, instances, first, base_vertex, base_instance);
}

void WebGPUCommandContext::CmdClearImage(Image *image, const ClearValue &value) {
  if (render_)
    throw std::runtime_error("clear inside render pass");
  // Consumed by the next pass that renders to the image, or by FlushClears.
  pending_clears_.insert_or_assign(image, value);
}

void WebGPUCommandContext::CmdCopyBuffer(Buffer *dst,
                                         Buffer *src,
                                         uint64_t size,
                                         uint64_t dst_offset,
                                         uint64_t src_offset) {
  if (render_)
    throw std::runtime_error("copy inside render pass");
  if (dst_offset > dst->Size() || size > dst->Size() - dst_offset || src_offset > src->Size() ||
      size > src->Size() - src_offset)
    throw std::out_of_range("WebGPU buffer copy");
  EndPasses();
  FlushClears();
  encoder_.CopyBufferToBuffer(Native(src)->Handle(), src_offset, Native(dst)->Handle(), dst_offset,
                              (size + 3) & ~uint64_t(3));
}

void WebGPUCommandContext::CmdPresent(Window *, Image *) {
  throw std::runtime_error("WebGPU hosts present images to their canvas");
}

wgpu::CommandBuffer WebGPUCommandContext::Finish() {
  EndPasses();
  FlushClears();
  return encoder_.Finish();
}

}  // namespace grassland::graphics::backend
